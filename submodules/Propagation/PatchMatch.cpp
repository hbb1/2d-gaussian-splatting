#include "PatchMatch.h"
#include <torch/extension.h>
#include <cfloat>

#include <cstdarg>

void StringAppendV(std::string* dst, const char* format, va_list ap) {
  // First try with a small fixed size buffer.
  static const int kFixedBufferSize = 1024;
  char fixed_buffer[kFixedBufferSize];

  // It is possible for methods that use a va_list to invalidate
  // the data in it upon use.  The fix is to make a copy
  // of the structure before using it and use that copy instead.
  va_list backup_ap;
  va_copy(backup_ap, ap);
  int result = vsnprintf(fixed_buffer, kFixedBufferSize, format, backup_ap);
  va_end(backup_ap);

  if (result < kFixedBufferSize) {
    if (result >= 0) {
      // Normal case - everything fits.
      dst->append(fixed_buffer, result);
      return;
    }

#ifdef _MSC_VER
    // Error or MSVC running out of space.  MSVC 8.0 and higher
    // can be asked about space needed with the special idiom below:
    va_copy(backup_ap, ap);
    result = vsnprintf(nullptr, 0, format, backup_ap);
    va_end(backup_ap);
#endif

    if (result < 0) {
      // Just an error.
      return;
    }
  }

  // Increase the buffer size to the size requested by vsnprintf,
  // plus one for the closing \0.
  const int variable_buffer_size = result + 1;
  std::unique_ptr<char> variable_buffer(new char[variable_buffer_size]);

  // Restore the va_list before we use it again.
  va_copy(backup_ap, ap);
  result =
      vsnprintf(variable_buffer.get(), variable_buffer_size, format, backup_ap);
  va_end(backup_ap);

  if (result >= 0 && result < variable_buffer_size) {
    dst->append(variable_buffer.get(), result);
  }
}

std::string StringPrintf(const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  std::string result;
  StringAppendV(&result, format, ap);
  va_end(ap);
  return result;
}

void CudaSafeCall(const cudaError_t error, const std::string& file,
                  const int line) {
  if (error != cudaSuccess) {
    std::cerr << StringPrintf("%s in %s at line %i", cudaGetErrorString(error),
                              file.c_str(), line)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

void CudaCheckError(const char* file, const int line) {
  cudaError error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << StringPrintf("cudaCheckError() failed at %s:%i : %s", file,
                              line, cudaGetErrorString(error))
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  error = cudaDeviceSynchronize();
  if (cudaSuccess != error) {
    std::cerr << StringPrintf("cudaCheckError() with sync failed at %s:%i : %s",
                              file, line, cudaGetErrorString(error))
              << std::endl;
    std::cerr
        << "This error is likely caused by the graphics card timeout "
           "detection mechanism of your operating system. Please refer to "
           "the FAQ in the documentation on how to solve this problem."
        << std::endl;
    exit(EXIT_FAILURE);
  }
}

PatchMatch::PatchMatch() {}

PatchMatch::~PatchMatch()
{
    delete[] plane_hypotheses_host;
    delete[] costs_host;

    for (int i = 0; i < num_images; ++i) {
        cudaDestroyTextureObject(texture_objects_host.images[i]);
        cudaFreeArray(cuArray[i]);
    }
    cudaFree(texture_objects_cuda);
    cudaFree(cameras_cuda);
    cudaFree(plane_hypotheses_cuda);
    cudaFree(costs_cuda);
    cudaFree(rand_states_cuda);
    cudaFree(selected_views_cuda);
    cudaFree(depths_cuda); 

    if (params.geom_consistency) {
        for (int i = 0; i < num_images; ++i) {
            cudaDestroyTextureObject(texture_depths_host.images[i]);
            cudaFreeArray(cuDepthArray[i]);
        }
        cudaFree(texture_depths_cuda);
    }
}

Camera ReadCamera(torch::Tensor intrinsic, torch::Tensor pose, torch::Tensor depth_interval)
{
    Camera camera;

    for (int i = 0; i < 3; ++i) {
        camera.R[3 * i + 0] = pose[i][0].item<float>();
        camera.R[3 * i + 1] = pose[i][1].item<float>();
        camera.R[3 * i + 2] = pose[i][2].item<float>();
        camera.t[i] = pose[i][3].item<float>();
    }

    for (int i = 0; i < 3; ++i) {
        camera.K[3 * i + 0] = intrinsic[i][0].item<float>();
        camera.K[3 * i + 1] = intrinsic[i][1].item<float>();
        camera.K[3 * i + 2] = intrinsic[i][2].item<float>();
    }

    camera.depth_min = depth_interval[0].item<float>();
    camera.depth_max = depth_interval[3].item<float>();

    return camera;
}

void RescaleImageAndCamera(torch::Tensor &src, torch::Tensor &dst, torch::Tensor &depth, Camera &camera)
{
    const int cols = depth.size(1);
    const int rows = depth.size(0);

    if (cols == src.size(1) && rows == src.size(0)) {
        dst = src.clone();
        return;
    }

    const float scale_x = cols / static_cast<float>(src.size(1));
    const float scale_y = rows / static_cast<float>(src.size(0));
    dst = torch::nn::functional::interpolate(src.unsqueeze(0), torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({rows, cols})).mode(torch::kBilinear)).squeeze(0);
    
    camera.K[0] *= scale_x;
    camera.K[2] *= scale_x;
    camera.K[4] *= scale_y;
    camera.K[5] *= scale_y;
    camera.width = cols;
    camera.height = rows;
}

float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera)
{
    float3 pointX;
    float3 tmpX;
    // Reprojection
    pointX.x = depth * (x - camera.K[2]) / camera.K[0];
    pointX.y = depth * (y - camera.K[5]) / camera.K[4];
    pointX.z = depth;

    // Rotation
    tmpX.x = camera.R[0] * pointX.x + camera.R[3] * pointX.y + camera.R[6] * pointX.z;
    tmpX.y = camera.R[1] * pointX.x + camera.R[4] * pointX.y + camera.R[7] * pointX.z;
    tmpX.z = camera.R[2] * pointX.x + camera.R[5] * pointX.y + camera.R[8] * pointX.z;

    // Transformation
    float3 C;
    C.x = -(camera.R[0] * camera.t[0] + camera.R[3] * camera.t[1] + camera.R[6] * camera.t[2]);
    C.y = -(camera.R[1] * camera.t[0] + camera.R[4] * camera.t[1] + camera.R[7] * camera.t[2]);
    C.z = -(camera.R[2] * camera.t[0] + camera.R[5] * camera.t[1] + camera.R[8] * camera.t[2]);
    pointX.x = tmpX.x + C.x;
    pointX.y = tmpX.y + C.y;
    pointX.z = tmpX.z + C.z;

    return pointX;
}

void ProjectonCamera(const float3 PointX, const Camera camera, float2 &point, float &depth)
{
    float3 tmp;
    tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
    tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
    tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

    depth = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
    point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
    point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;
}

float GetAngle(const torch::Tensor &v1, const torch::Tensor &v2)
{
    float dot_product = v1[0].item<float>() * v2[0].item<float>() + v1[1].item<float>() * v2[1].item<float>() + v1[2].item<float>() * v2[2].item<float>();
    float angle = acosf(dot_product);
    //if angle is not a number the dot product was 1 and thus the two vectors should be identical --> return 0
    if ( angle != angle )
        return 0.0f;

    return angle;
}

void StoreColorPlyFileBinaryPointCloud (const std::string &plyFilePath, const std::vector<PointList> &pc)
{
    std::cout << "store 3D points to ply file" << std::endl;

    FILE *outputPly;
    outputPly=fopen(plyFilePath.c_str(), "wb");

    /*write header*/
    fprintf(outputPly, "ply\n");
    fprintf(outputPly, "format binary_little_endian 1.0\n");
    fprintf(outputPly, "element vertex %d\n",pc.size());
    fprintf(outputPly, "property float x\n");
    fprintf(outputPly, "property float y\n");
    fprintf(outputPly, "property float z\n");
    fprintf(outputPly, "property float nx\n");
    fprintf(outputPly, "property float ny\n");
    fprintf(outputPly, "property float nz\n");
    fprintf(outputPly, "property uchar red\n");
    fprintf(outputPly, "property uchar green\n");
    fprintf(outputPly, "property uchar blue\n");
    fprintf(outputPly, "end_header\n");

    //write data
#pragma omp parallel for
    for(size_t i = 0; i < pc.size(); i++) {
        const PointList &p = pc[i];
        float3 X = p.coord;
        const float3 normal = p.normal;
        const float3 color = p.color;
        const char b_color = (int)color.x;
        const char g_color = (int)color.y;
        const char r_color = (int)color.z;

        if(!(X.x < FLT_MAX && X.x > -FLT_MAX) || !(X.y < FLT_MAX && X.y > -FLT_MAX) || !(X.z < FLT_MAX && X.z >= -FLT_MAX)){
            X.x = 0.0f;
            X.y = 0.0f;
            X.z = 0.0f;
        }
#pragma omp critical
        {
            fwrite(&X.x,      sizeof(X.x), 1, outputPly);
            fwrite(&X.y,      sizeof(X.y), 1, outputPly);
            fwrite(&X.z,      sizeof(X.z), 1, outputPly);
            fwrite(&normal.x, sizeof(normal.x), 1, outputPly);
            fwrite(&normal.y, sizeof(normal.y), 1, outputPly);
            fwrite(&normal.z, sizeof(normal.z), 1, outputPly);
            fwrite(&r_color,  sizeof(char), 1, outputPly);
            fwrite(&g_color,  sizeof(char), 1, outputPly);
            fwrite(&b_color,  sizeof(char), 1, outputPly);
        }

    }
    fclose(outputPly);
}

static float GetDisparity(const Camera &camera, const int2 &p, const float &depth)
{
    float point3D[3];
    point3D[0] = depth * (p.x - camera.K[2]) / camera.K[0];
    point3D[1] = depth * (p.y - camera.K[5]) / camera.K[4];
    point3D[2] = depth;

    return std::sqrt(point3D[0] * point3D[0] + point3D[1] * point3D[1] + point3D[2] * point3D[2]);
}

void PatchMatch::SetGeomConsistencyParams()
{
    params.geom_consistency = true;
    params.max_iterations = 2;
}

void PatchMatch::InuputInitialization(torch::Tensor images_cuda, torch::Tensor intrinsics_cuda, torch::Tensor poses_cuda, 
                                torch::Tensor depth_cuda, torch::Tensor normal_cuda, torch::Tensor depth_intervals)
{
    images.clear();
    cameras.clear();

    torch::Tensor image_color = images_cuda[0];
    torch::Tensor image_float = torch::mean(image_color, /*dim=*/2, /*keepdim=*/true).squeeze();
    image_float = image_float.to(torch::kFloat32);
    images.push_back(image_float);

    Camera camera = ReadCamera(intrinsics_cuda[0], poses_cuda[0], depth_intervals[0]);
    camera.height = image_float.size(0);
    camera.width = image_float.size(1);
    cameras.push_back(camera);

    torch::Tensor ref_depth = depth_cuda;
    depths.push_back(ref_depth);

    torch::Tensor ref_normal = normal_cuda;
    normals.push_back(ref_normal);

    int num_src_images = images_cuda.size(0);
    for (int i = 1; i < num_src_images; ++i) {
        torch::Tensor src_image_color = images_cuda[i];
        torch::Tensor src_image_float = torch::mean(src_image_color, /*dim=*/2, /*keepdim=*/true).squeeze();
        src_image_float = src_image_float.to(torch::kFloat32);
        images.push_back(src_image_float);

        Camera camera = ReadCamera(intrinsics_cuda[i], poses_cuda[i], depth_intervals[i]);
        camera.height = src_image_float.size(0);
        camera.width = src_image_float.size(1);
        cameras.push_back(camera);
    }

    // Scale cameras and images
    for (size_t i = 0; i < images.size(); ++i) {
        if (images[i].size(1) <= params.max_image_size && images[i].size(0) <= params.max_image_size) {
            continue;
        }

        const float factor_x = static_cast<float>(params.max_image_size) / images[i].size(1);
        const float factor_y = static_cast<float>(params.max_image_size) / images[i].size(0);
        const float factor = std::min(factor_x, factor_y);

        const int new_cols = std::round(images[i].size(1) * factor);
        const int new_rows = std::round(images[i].size(0) * factor);

        const float scale_x = new_cols / static_cast<float>(images[i].size(1));
        const float scale_y = new_rows / static_cast<float>(images[i].size(0));

        torch::Tensor scaled_image_float = torch::nn::functional::interpolate(images[i].unsqueeze(0), torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({new_rows, new_cols})).mode(torch::kBilinear)).squeeze(0);
        images[i] = scaled_image_float.clone();

        cameras[i].K[0] *= scale_x;
        cameras[i].K[2] *= scale_x;
        cameras[i].K[4] *= scale_y;
        cameras[i].K[5] *= scale_y;
        cameras[i].height = scaled_image_float.size(0);
        cameras[i].width = scaled_image_float.size(1);
    }

    params.depth_min = cameras[0].depth_min * 0.6f;
    params.depth_max = cameras[0].depth_max * 1.2f;
    params.num_images = (int)images.size();
    params.disparity_min = cameras[0].K[0] * params.baseline / params.depth_max;
    params.disparity_max = cameras[0].K[0] * params.baseline / params.depth_min;

}

void PatchMatch::CudaSpaceInitialization()
{
    num_images = (int)images.size();

    for (int i = 0; i < num_images; ++i) {
        int rows = images[i].size(0);
        int cols = images[i].size(1);

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        cudaMallocArray(&cuArray[i], &channelDesc, cols, rows);

        cudaMemcpy2DToArray(cuArray[i], 0, 0, images[i].data_ptr<float>(), images[i].stride(0) * sizeof(float), cols * sizeof(float), rows, cudaMemcpyHostToDevice);

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaCreateTextureObject(&(texture_objects_host.images[i]), &resDesc, &texDesc, NULL);
    }

    cudaMalloc((void**)&texture_objects_cuda, sizeof(cudaTextureObjects));
    cudaMemcpy(texture_objects_cuda, &texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cameras_cuda, sizeof(Camera) * (num_images));
    cudaMemcpy(cameras_cuda, &cameras[0], sizeof(Camera) * (num_images), cudaMemcpyHostToDevice);

    int total_pixels = cameras[0].height * cameras[0].width;    
    // Concatenate normals and depths into a single tensor
    torch::Tensor plane_hypotheses_tensor = torch::cat({
        normals[0].view({total_pixels, 3}),
        depths[0].view({total_pixels, 1})
    }, 1);

    // TODO: Fix initialization bug
    // auto plane_hypotheses_float4 = plane_hypotheses_tensor.to(torch::kFloat32).view({-1, 4});
    // plane_hypotheses_host = reinterpret_cast<float4*>(plane_hypotheses_float4.data_ptr<float>());
    plane_hypotheses_host = new float4[total_pixels];
    cudaMalloc((void**)&plane_hypotheses_cuda, sizeof(float4) * total_pixels);
    cudaMemcpy(plane_hypotheses_cuda, plane_hypotheses_host, sizeof(float4) * total_pixels, cudaMemcpyHostToDevice);

    costs_host = new float[cameras[0].height * cameras[0].width];
    cudaMalloc((void**)&costs_cuda, sizeof(float) * (cameras[0].height * cameras[0].width));

    cudaMalloc((void**)&rand_states_cuda, sizeof(curandState) * (cameras[0].height * cameras[0].width));
    cudaMalloc((void**)&selected_views_cuda, sizeof(unsigned int) * (cameras[0].height * cameras[0].width));

    cudaMalloc((void**)&depths_cuda, sizeof(float) * (cameras[0].height * cameras[0].width));
    cudaMemcpy(depths_cuda, depths[0].data_ptr<float>(), sizeof(float) * cameras[0].height * cameras[0].width, cudaMemcpyHostToDevice);
}

int PatchMatch::GetReferenceImageWidth()
{
    return cameras[0].width;
}

int PatchMatch::GetReferenceImageHeight()
{
    return cameras[0].height;
}

torch::Tensor PatchMatch::GetReferenceImage()
{
    return images[0];
}

float4 PatchMatch::GetPlaneHypothesis(const int index)
{
    return plane_hypotheses_host[index];
}

float4* PatchMatch::GetPlaneHypotheses()
{
    return plane_hypotheses_host;
}

float PatchMatch::GetCost(const int index)
{
    return costs_host[index];
}

void PatchMatch::SetPatchSize(int patch_size)
{
    params.patch_size = patch_size;
}

int PatchMatch::GetPatchSize()
{
    return params.patch_size;
}


