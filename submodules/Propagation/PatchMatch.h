#ifndef _PatchMatch_H_
#define _PatchMatch_H_

#include "main.h"
#include <torch/extension.h>

Camera ReadCamera(torch::Tensor intrinsic, torch::Tensor pose, torch::Tensor depth_interval);
void RescaleImageAndCamera(torch::Tensor &src, torch::Tensor &dst, torch::Tensor &depth, Camera &camera);
float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera);
void ProjectonCamera(const float3 PointX, const Camera camera, float2 &point, float &depth);
float GetAngle(const torch::Tensor &v1, const torch::Tensor &v2);
void StoreColorPlyFileBinaryPointCloud(const std::string &plyFilePath, const std::vector<PointList> &pc);

#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR() CudaCheckError(__FILE__, __LINE__)

void CudaSafeCall(const cudaError_t error, const std::string& file, const int line);
void CudaCheckError(const char* file, const int line);

struct cudaTextureObjects {
    cudaTextureObject_t images[MAX_IMAGES];
};

struct PatchMatchParams {
    int max_iterations = 6;
    int patch_size = 11;
    int num_images = 5;
    int max_image_size=3200;
    int radius_increment = 2;
    float sigma_spatial = 5.0f;
    float sigma_color = 3.0f;
    int top_k = 4;
    float baseline = 0.54f;
    float depth_min = 0.0f;
    float depth_max = 1.0f;
    float disparity_min = 0.0f;
    float disparity_max = 1.0f;
    bool geom_consistency = false;
};

class PatchMatch {
public:
    PatchMatch();
    ~PatchMatch();

    void InuputInitialization(torch::Tensor images_cuda, torch::Tensor intrinsics_cuda, torch::Tensor poses_cuda, torch::Tensor depth_cuda, torch::Tensor normal_cuda, torch::Tensor depth_intervals);
    void Colmap2MVS(const std::string &dense_folder, std::vector<Problem> &problems);
    void CudaSpaceInitialization();
    void RunPatchMatch();
    void SetGeomConsistencyParams();
    void SetPatchSize(int patch_size);
    int GetPatchSize();
    int GetReferenceImageWidth();
    int GetReferenceImageHeight();
    torch::Tensor GetReferenceImage();
    float4 GetPlaneHypothesis(const int index);
    float GetCost(const int index);
    float4* GetPlaneHypotheses();

private:
    int num_images;
    std::vector<torch::Tensor> images;
    std::vector<torch::Tensor> depths;
    std::vector<torch::Tensor> normals;
    std::vector<Camera> cameras;
    cudaTextureObjects texture_objects_host;
    cudaTextureObjects texture_depths_host;
    float4 *plane_hypotheses_host;
    float *costs_host;
    PatchMatchParams params;

    Camera *cameras_cuda;
    cudaArray *cuArray[MAX_IMAGES];
    cudaArray *cuDepthArray[MAX_IMAGES];
    cudaTextureObjects *texture_objects_cuda;
    cudaTextureObjects *texture_depths_cuda;
    float4 *plane_hypotheses_cuda;
    float *costs_cuda;
    curandState *rand_states_cuda;
    unsigned int *selected_views_cuda;
    float *depths_cuda;
};

#endif // _PatchMatch_H_
