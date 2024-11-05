#include "PatchMatch.h"
#include <torch/extension.h>
#include <chrono>

__device__  void sort_small(float *d, const int n)
{
    int j;
    for (int i = 1; i < n; i++) {
        float tmp = d[i];
        for (j = i; j >= 1 && tmp < d[j-1]; j--)
            d[j] = d[j-1];
        d[j] = tmp;
    }
}

__device__ void sort_small_weighted(float *d, float *w, int n)
{
    int j;
    for (int i = 1; i < n; i++) {
        float tmp = d[i];
        float tmp_w = w[i];
        for (j = i; j >= 1 && tmp < d[j - 1]; j--) {
            d[j] = d[j - 1];
            w[j] = w[j - 1];
        }
        d[j] = tmp;
        w[j] = tmp_w;
    }
}

__device__ int FindMinCostIndex(const float *costs, const int n)
{
    float min_cost = costs[0];
    int min_cost_idx = 0;
    for (int idx = 1; idx < n; ++idx) {
        if (costs[idx] <= min_cost) {
            min_cost = costs[idx];
            min_cost_idx = idx;
        }
    }
    return min_cost_idx;
}

__device__  void setBit(unsigned int &input, const unsigned int n)
{
    input |= (unsigned int)(1 << n);
}

__device__  int isSet(unsigned int input, const unsigned int n)
{
    return (input >> n) & 1;
}

__device__ void Mat33DotVec3(const float mat[9], const float4 vec, float4 *result)
{
  result->x = mat[0] * vec.x + mat[1] * vec.y + mat[2] * vec.z;
  result->y = mat[3] * vec.x + mat[4] * vec.y + mat[5] * vec.z;
  result->z = mat[6] * vec.x + mat[7] * vec.y + mat[8] * vec.z;
}

__device__ float Vec3DotVec3(const float4 vec1, const float4 vec2)
{
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

__device__ void NormalizeVec3 (float4 *vec)
{
    const float normSquared = vec->x * vec->x + vec->y * vec->y + vec->z * vec->z;
    const float inverse_sqrt = rsqrtf (normSquared);
    vec->x *= inverse_sqrt;
    vec->y *= inverse_sqrt;
    vec->z *= inverse_sqrt;
}

__device__ void TransformPDFToCDF(float* probs, const int num_probs)
{
    float prob_sum = 0.0f;
    for (int i = 0; i < num_probs; ++i) {
        prob_sum += probs[i];
    }
    const float inv_prob_sum = 1.0f / prob_sum;

    float cum_prob = 0.0f;
    for (int i = 0; i < num_probs; ++i) {
        const float prob = probs[i] * inv_prob_sum;
        cum_prob += prob;
        probs[i] = cum_prob;
    }
}

__device__ void Get3DPoint(const Camera camera, const int2 p, const float depth, float *X)
{
    X[0] = depth * (p.x - camera.K[2]) / camera.K[0];
    X[1] = depth * (p.y - camera.K[5]) / camera.K[4];
    X[2] = depth;
}

__device__ float4 GetViewDirection(const Camera camera, const int2 p, const float depth)
{
    float X[3];
    Get3DPoint(camera, p, depth, X);
    float norm = sqrt(X[0] * X[0] + X[1] * X[1] + X[2] * X[2]);

    float4 view_direction;
    view_direction.x = X[0] / norm;
    view_direction.y = X[1] / norm;
    view_direction.z =  X[2] / norm;
    view_direction.w = 0;
    return view_direction;
}

__device__ float GetDistance2Origin(const Camera camera, const int2 p, const float depth, const float4 normal)
{
    float X[3];
    Get3DPoint(camera, p, depth, X);
    return -(normal.x * X[0] + normal.y * X[1] + normal.z * X[2]);
}

__device__ float ComputeDepthfromPlaneHypothesis(const Camera camera, const float4 plane_hypothesis, const int2 p)
{
    return -plane_hypothesis.w * camera.K[0] / ((p.x - camera.K[2]) * plane_hypothesis.x + (camera.K[0] / camera.K[4]) * (p.y - camera.K[5]) * plane_hypothesis.y + camera.K[0] * plane_hypothesis.z);
}

__device__ float4 GenerateRandomNormal(const Camera camera, const int2 p, curandState *rand_state, const float depth)
{
    float4 normal;
    float q1 = 1.0f;
    float q2 = 1.0f;
    float s = 2.0f;
    while (s >= 1.0f) {
        q1 = 2.0f * curand_uniform(rand_state) -1.0f;
        q2 = 2.0f * curand_uniform(rand_state) - 1.0f;
        s = q1 * q1 + q2 * q2;
    }
    const float sq = sqrt(1.0f - s);
    normal.x = 2.0f * q1 * sq;
    normal.y = 2.0f * q2 * sq;
    normal.z = 1.0f - 2.0f * s;
    normal.w = 0;

    float4 view_direction = GetViewDirection(camera, p, depth);
    float dot_product = normal.x * view_direction.x + normal.y * view_direction.y + normal.z * view_direction.z;
    if (dot_product > 0.0f) {
        normal.x = -normal.x;
        normal.y = -normal.y;
        normal.z = - normal.z;
    }
    NormalizeVec3(&normal);
    return normal;
}

__device__ float4 GeneratePerturbedNormal(const Camera camera, const int2 p, const float4 normal, curandState *rand_state, const float perturbation)
{
    float4 view_direction = GetViewDirection(camera, p, 1.0f);

    const float a1 = (curand_uniform(rand_state) - 0.5f) * perturbation;
    const float a2 = (curand_uniform(rand_state) - 0.5f) * perturbation;
    const float a3 = (curand_uniform(rand_state) - 0.5f) * perturbation;

    const float sin_a1 = sin(a1);
    const float sin_a2 = sin(a2);
    const float sin_a3 = sin(a3);
    const float cos_a1 = cos(a1);
    const float cos_a2 = cos(a2);
    const float cos_a3 = cos(a3);

    float R[9];
    R[0] = cos_a2 * cos_a3;
    R[1] = cos_a3 * sin_a1 * sin_a2 - cos_a1 * sin_a3;
    R[2] = sin_a1 * sin_a3 + cos_a1 * cos_a3 * sin_a2;
    R[3] = cos_a2 * sin_a3;
    R[4] = cos_a1 * cos_a3 + sin_a1 * sin_a2 * sin_a3;
    R[5] = cos_a1 * sin_a2 * sin_a3 - cos_a3 * sin_a1;
    R[6] = -sin_a2;
    R[7] = cos_a2 * sin_a1;
    R[8] = cos_a1 * cos_a2;

    float4 normal_perturbed;
    Mat33DotVec3(R, normal, &normal_perturbed);

    if (Vec3DotVec3(normal_perturbed, view_direction) >= 0.0f) {
        normal_perturbed = normal;
    }

    NormalizeVec3(&normal_perturbed);
    return normal_perturbed;
}

__device__ float4 GenerateRandomPlaneHypothesis(const Camera camera, const int2 p, curandState *rand_state, const float depth_min, const float depth_max, float init_depth)
{
    float depth = init_depth;
    if (depth <= 0){
        depth = curand_uniform(rand_state) * (depth_max - depth_min) + depth_min;
    }
    // printf("initdepth: %f\n", init_depth);

    float4 plane_hypothesis = GenerateRandomNormal(camera, p, rand_state, depth);
    plane_hypothesis.w = GetDistance2Origin(camera, p, depth, plane_hypothesis);
    return plane_hypothesis;
}

__device__ float4 GeneratePertubedPlaneHypothesis(const Camera camera, const int2 p, curandState *rand_state, const float perturbation, const float4 plane_hypothesis_now, const float depth_now, const float depth_min, const float depth_max)
{
    float depth_perturbed = depth_now;

    float dist_perturbed = plane_hypothesis_now.w;
    const float dist_min_perturbed = (1 - perturbation) * dist_perturbed;
    const float dist_max_perturbed = (1 + perturbation) * dist_perturbed;
    float4 plane_hypothesis_temp = plane_hypothesis_now;
    do {
        dist_perturbed = curand_uniform(rand_state) * (dist_max_perturbed - dist_min_perturbed) + dist_min_perturbed;
        plane_hypothesis_temp.w = dist_perturbed;
        depth_perturbed = ComputeDepthfromPlaneHypothesis(camera, plane_hypothesis_temp, p);
    } while (depth_perturbed < depth_min && depth_perturbed > depth_max);

    float4 plane_hypothesis = GeneratePerturbedNormal(camera, p, plane_hypothesis_now, rand_state, perturbation * M_PI);
    plane_hypothesis.w = dist_perturbed;
    return plane_hypothesis;
}

__device__ void ComputeHomography(const Camera ref_camera, const Camera src_camera, const float4 plane_hypothesis, float *H)
{
    float ref_C[3];
    float src_C[3];
    ref_C[0] = -(ref_camera.R[0] * ref_camera.t[0] + ref_camera.R[3] * ref_camera.t[1] + ref_camera.R[6] * ref_camera.t[2]);
    ref_C[1] = -(ref_camera.R[1] * ref_camera.t[0] + ref_camera.R[4] * ref_camera.t[1] + ref_camera.R[7] * ref_camera.t[2]);
    ref_C[2] = -(ref_camera.R[2] * ref_camera.t[0] + ref_camera.R[5] * ref_camera.t[1] + ref_camera.R[8] * ref_camera.t[2]);
    src_C[0] = -(src_camera.R[0] * src_camera.t[0] + src_camera.R[3] * src_camera.t[1] + src_camera.R[6] * src_camera.t[2]);
    src_C[1] = -(src_camera.R[1] * src_camera.t[0] + src_camera.R[4] * src_camera.t[1] + src_camera.R[7] * src_camera.t[2]);
    src_C[2] = -(src_camera.R[2] * src_camera.t[0] + src_camera.R[5] * src_camera.t[1] + src_camera.R[8] * src_camera.t[2]);

    float R_relative[9];
    float C_relative[3];
    float t_relative[3];
    R_relative[0] = src_camera.R[0] * ref_camera.R[0] + src_camera.R[1] * ref_camera.R[1] + src_camera.R[2] *ref_camera.R[2];
    R_relative[1] = src_camera.R[0] * ref_camera.R[3] + src_camera.R[1] * ref_camera.R[4] + src_camera.R[2] *ref_camera.R[5];
    R_relative[2] = src_camera.R[0] * ref_camera.R[6] + src_camera.R[1] * ref_camera.R[7] + src_camera.R[2] *ref_camera.R[8];
    R_relative[3] = src_camera.R[3] * ref_camera.R[0] + src_camera.R[4] * ref_camera.R[1] + src_camera.R[5] *ref_camera.R[2];
    R_relative[4] = src_camera.R[3] * ref_camera.R[3] + src_camera.R[4] * ref_camera.R[4] + src_camera.R[5] *ref_camera.R[5];
    R_relative[5] = src_camera.R[3] * ref_camera.R[6] + src_camera.R[4] * ref_camera.R[7] + src_camera.R[5] *ref_camera.R[8];
    R_relative[6] = src_camera.R[6] * ref_camera.R[0] + src_camera.R[7] * ref_camera.R[1] + src_camera.R[8] *ref_camera.R[2];
    R_relative[7] = src_camera.R[6] * ref_camera.R[3] + src_camera.R[7] * ref_camera.R[4] + src_camera.R[8] *ref_camera.R[5];
    R_relative[8] = src_camera.R[6] * ref_camera.R[6] + src_camera.R[7] * ref_camera.R[7] + src_camera.R[8] *ref_camera.R[8];
    C_relative[0] = (ref_C[0] - src_C[0]);
    C_relative[1] = (ref_C[1] - src_C[1]);
    C_relative[2] = (ref_C[2] - src_C[2]);
    t_relative[0] = src_camera.R[0] * C_relative[0] + src_camera.R[1] * C_relative[1] + src_camera.R[2] * C_relative[2];
    t_relative[1] = src_camera.R[3] * C_relative[0] + src_camera.R[4] * C_relative[1] + src_camera.R[5] * C_relative[2];
    t_relative[2] = src_camera.R[6] * C_relative[0] + src_camera.R[7] * C_relative[1] + src_camera.R[8] * C_relative[2];

    H[0] = R_relative[0] - t_relative[0] * plane_hypothesis.x / plane_hypothesis.w;
    H[1] = R_relative[1] - t_relative[0] * plane_hypothesis.y / plane_hypothesis.w;
    H[2] = R_relative[2] - t_relative[0] * plane_hypothesis.z / plane_hypothesis.w;
    H[3] = R_relative[3] - t_relative[1] * plane_hypothesis.x / plane_hypothesis.w;
    H[4] = R_relative[4] - t_relative[1] * plane_hypothesis.y / plane_hypothesis.w;
    H[5] = R_relative[5] - t_relative[1] * plane_hypothesis.z / plane_hypothesis.w;
    H[6] = R_relative[6] - t_relative[2] * plane_hypothesis.x / plane_hypothesis.w;
    H[7] = R_relative[7] - t_relative[2] * plane_hypothesis.y / plane_hypothesis.w;
    H[8] = R_relative[8] - t_relative[2] * plane_hypothesis.z / plane_hypothesis.w;

    float tmp[9];
    tmp[0] = H[0] / ref_camera.K[0];
    tmp[1] = H[1] / ref_camera.K[4];
    tmp[2] = -H[0] * ref_camera.K[2] / ref_camera.K[0] - H[1] * ref_camera.K[5] / ref_camera.K[4] + H[2];
    tmp[3] = H[3] / ref_camera.K[0];
    tmp[4] = H[4] / ref_camera.K[4];
    tmp[5] = -H[3] * ref_camera.K[2] / ref_camera.K[0] - H[4] * ref_camera.K[5] / ref_camera.K[4] + H[5];
    tmp[6] = H[6] / ref_camera.K[0];
    tmp[7] = H[7] / ref_camera.K[4];
    tmp[8] = -H[6] * ref_camera.K[2] / ref_camera.K[0] - H[7] * ref_camera.K[5] / ref_camera.K[4] + H[8];

    H[0] = src_camera.K[0] * tmp[0] + src_camera.K[2] * tmp[6];
    H[1] = src_camera.K[0] * tmp[1] + src_camera.K[2] * tmp[7];
    H[2] = src_camera.K[0] * tmp[2] + src_camera.K[2] * tmp[8];
    H[3] = src_camera.K[4] * tmp[3] + src_camera.K[5] * tmp[6];
    H[4] = src_camera.K[4] * tmp[4] + src_camera.K[5] * tmp[7];
    H[5] = src_camera.K[4] * tmp[5] + src_camera.K[5] * tmp[8];
    H[6] = src_camera.K[8] * tmp[6];
    H[7] = src_camera.K[8] * tmp[7];
    H[8] = src_camera.K[8] * tmp[8];
}

__device__ float2 ComputeCorrespondingPoint(const float *H, const int2 p)
{
    float3 pt;
    pt.x = H[0] * p.x + H[1] * p.y + H[2];
    pt.y = H[3] * p.x + H[4] * p.y + H[5];
    pt.z = H[6] * p.x + H[7] * p.y + H[8];
    return make_float2(pt.x / pt.z, pt.y / pt.z);
}

__device__ float4 TransformNormal(const Camera camera, float4 plane_hypothesis)
{
    float4 transformed_normal;
    transformed_normal.x = camera.R[0] * plane_hypothesis.x + camera.R[3] * plane_hypothesis.y + camera.R[6] * plane_hypothesis.z;
    transformed_normal.y = camera.R[1] * plane_hypothesis.x + camera.R[4] * plane_hypothesis.y + camera.R[7] * plane_hypothesis.z;
    transformed_normal.z = camera.R[2] * plane_hypothesis.x + camera.R[5] * plane_hypothesis.y + camera.R[8] * plane_hypothesis.z;
    transformed_normal.w = plane_hypothesis.w;
    return transformed_normal;
}

__device__ float4 TransformNormal2RefCam(const Camera camera, float4 plane_hypothesis)
{
    float4 transformed_normal;
    transformed_normal.x = camera.R[0] * plane_hypothesis.x + camera.R[1] * plane_hypothesis.y + camera.R[2] * plane_hypothesis.z;
    transformed_normal.y = camera.R[3] * plane_hypothesis.x + camera.R[4] * plane_hypothesis.y + camera.R[5] * plane_hypothesis.z;
    transformed_normal.z = camera.R[6] * plane_hypothesis.x + camera.R[7] * plane_hypothesis.y + camera.R[8] * plane_hypothesis.z;
    transformed_normal.w = plane_hypothesis.w;
    return transformed_normal;
}

__device__ float ComputeBilateralWeight(const float x_dist, const float y_dist, const float pix, const float center_pix, const float sigma_spatial, const float sigma_color)
{
    const float spatial_dist = sqrt(x_dist * x_dist + y_dist * y_dist);
    const float color_dist = fabs(pix - center_pix);
    return exp(-spatial_dist / (2.0f * sigma_spatial* sigma_spatial) - color_dist / (2.0f * sigma_color * sigma_color));
}

__device__ float ComputeBilateralNCC(const cudaTextureObject_t ref_image, const Camera ref_camera, const cudaTextureObject_t src_image, const Camera src_camera, const int2 p, const float4 plane_hypothesis, const PatchMatchParams params)
{
    const float cost_max = 2.0f;
    int radius = params.patch_size / 2;

    float H[9];
    ComputeHomography(ref_camera, src_camera, plane_hypothesis, H);
    float2 pt = ComputeCorrespondingPoint(H, p);
    if (pt.x >= src_camera.width || pt.x < 0.0f || pt.y >= src_camera.height || pt.y < 0.0f) {
        return cost_max;
    }

    float cost = 0.0f;
    {
        float sum_ref = 0.0f;
        float sum_ref_ref = 0.0f;
        float sum_src = 0.0f;
        float sum_src_src = 0.0f;
        float sum_ref_src = 0.0f;
        float bilateral_weight_sum = 0.0f;
        const float ref_center_pix = tex2D<float>(ref_image, p.x + 0.5f, p.y + 0.5f);

        for (int i = -radius; i < radius + 1; i += params.radius_increment) {
            float sum_ref_row = 0.0f;
            float sum_src_row = 0.0f;
            float sum_ref_ref_row = 0.0f;
            float sum_src_src_row = 0.0f;
            float sum_ref_src_row = 0.0f;
            float bilateral_weight_sum_row = 0.0f;

            for (int j = -radius; j < radius + 1; j += params.radius_increment) {
                const int2 ref_pt = make_int2(p.x + i, p.y + j);
                const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
                float2 src_pt = ComputeCorrespondingPoint(H, ref_pt);
                const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);

                float weight = ComputeBilateralWeight(i, j, ref_pix, ref_center_pix, params.sigma_spatial, params.sigma_color);

                sum_ref_row += weight * ref_pix;
                sum_ref_ref_row += weight * ref_pix * ref_pix;
                sum_src_row += weight * src_pix;
                sum_src_src_row += weight * src_pix * src_pix;
                sum_ref_src_row += weight * ref_pix * src_pix;
                bilateral_weight_sum_row += weight;
            }

            sum_ref += sum_ref_row;
            sum_ref_ref += sum_ref_ref_row;
            sum_src += sum_src_row;
            sum_src_src += sum_src_src_row;
            sum_ref_src += sum_ref_src_row;
            bilateral_weight_sum += bilateral_weight_sum_row;
        }
        const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
        sum_ref *= inv_bilateral_weight_sum;
        sum_ref_ref *= inv_bilateral_weight_sum;
        sum_src *= inv_bilateral_weight_sum;
        sum_src_src *= inv_bilateral_weight_sum;
        sum_ref_src *= inv_bilateral_weight_sum;

        const float var_ref = sum_ref_ref - sum_ref * sum_ref;
        const float var_src = sum_src_src - sum_src * sum_src;

        const float kMinVar = 1e-5f;
        if (var_ref < kMinVar || var_src < kMinVar) {
            return cost = cost_max;
        } else {
            const float covar_src_ref = sum_ref_src - sum_ref * sum_src;
            const float var_ref_src = sqrt(var_ref * var_src);
            return cost = max(0.0f, min(cost_max, 1.0f - covar_src_ref / var_ref_src));
        }
    }
}

__device__ float ComputeMultiViewInitialCostandSelectedViews(const cudaTextureObject_t *images, const Camera *cameras, const int2 p, const float4 plane_hypothesis, unsigned int *selected_views, const PatchMatchParams params)
{
    float cost_max = 2.0f;
    float cost_vector[32] = {2.0f};
    float cost_vector_copy[32] = {2.0f};
    int cost_count = 0;
    int num_valid_views = 0;

    for (int i = 1; i < params.num_images; ++i) {
        float c = ComputeBilateralNCC(images[0], cameras[0], images[i], cameras[i], p, plane_hypothesis, params);
        cost_vector[i - 1] = c;
        cost_vector_copy[i - 1] = c;
        cost_count++;
        if (c < cost_max) {
            num_valid_views++;
        }
    }

    sort_small(cost_vector, cost_count);
    *selected_views = 0;

    int top_k = min(num_valid_views, params.top_k);
    if (top_k > 0) {
        float cost = 0.0f;
        for (int i = 0; i < top_k; ++i) {
            cost += cost_vector[i];
        }
        float cost_threshold = cost_vector[top_k - 1];
        for (int i = 0; i < params.num_images - 1; ++i) {
            if (cost_vector_copy[i] <= cost_threshold) {
                setBit(*selected_views, i);
            }
        }
        return cost / top_k;
    } else {
        return cost_max;
    }
}

__device__ void ComputeMultiViewCostVector(const cudaTextureObject_t *images, const Camera *cameras, const int2 p, const float4 plane_hypothesis, float *cost_vector, const PatchMatchParams params)
{
    for (int i = 1; i < params.num_images; ++i) {
        cost_vector[i - 1] = ComputeBilateralNCC(images[0], cameras[0], images[i], cameras[i], p, plane_hypothesis, params);
    }
}

__device__ float3 Get3DPointonWorld_cu(const float x, const float y, const float depth, const Camera camera)
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

__device__ void ProjectonCamera_cu(const float3 PointX, const Camera camera, float2 &point, float &depth)
{
    float3 tmp;
    tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
    tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
    tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

    depth = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
    point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
    point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;
}

__device__ float ComputeGeomConsistencyCost(const cudaTextureObject_t depth_image, const Camera ref_camera, const Camera src_camera, const float4 plane_hypothesis, const int2 p)
{
    const float max_cost = 5.0f;

    float depth = ComputeDepthfromPlaneHypothesis(ref_camera, plane_hypothesis, p);
    float3 forward_point = Get3DPointonWorld_cu(p.x, p.y, depth, ref_camera);

    float2 src_pt;
    float src_d;
    ProjectonCamera_cu(forward_point, src_camera, src_pt, src_d);
    const float src_depth = tex2D<float>(depth_image,  (int)src_pt.x + 0.5f, (int)src_pt.y + 0.5f);

    if (src_depth == 0.0f) {
        return max_cost;
    }

    float3 src_3D_pt = Get3DPointonWorld_cu(src_pt.x, src_pt.y, src_depth, src_camera);

    float2 backward_point;
    float ref_d;
    ProjectonCamera_cu(src_3D_pt, ref_camera, backward_point, ref_d);

    const float diff_col = p.x - backward_point.x;
    const float diff_row = p.y - backward_point.y;
    return min(max_cost, sqrt(diff_col * diff_col + diff_row * diff_row));
}

__global__ void RandomInitialization(cudaTextureObjects *texture_objects, Camera *cameras, float4 *plane_hypotheses, 
                                     float *costs, curandState *rand_states, unsigned int *selected_views, 
                                     const PatchMatchParams params, float *depths)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    int width = cameras[0].width;
    int height = cameras[0].height;

    if (p.x >= width || p.y >= height) {
        return;
    }

    const int center = p.y * width + p.x;
    curand_init(clock64(), p.y, p.x, &rand_states[center]);

    float4 plane_hypothesis = plane_hypotheses[center];
    plane_hypothesis = TransformNormal2RefCam(cameras[0], plane_hypothesis);
    float depth = plane_hypothesis.w;
    plane_hypothesis.w = GetDistance2Origin(cameras[0], p, depth, plane_hypothesis);
    plane_hypotheses[center] = plane_hypothesis;
    costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images, cameras, p, plane_hypotheses[center], &selected_views[center], params);
}

__device__ void PlaneHypothesisRefinement(const cudaTextureObject_t *images, const cudaTextureObject_t *depth_images, const Camera *cameras, float4 *plane_hypothesis, float *depth, float *cost, curandState *rand_state, const float *view_weights, const float weight_norm, const int2 p, const PatchMatchParams params)
{
    float perturbation = 0.02f;

    float depth_rand = curand_uniform(rand_state) * (params.depth_max - params.depth_min) + params.depth_min;
    float depth_perturbed = *depth;
    const float depth_min_perturbed = (1 - perturbation) * depth_perturbed;
    const float depth_max_perturbed = (1 + perturbation) * depth_perturbed;
    do {
        depth_perturbed = curand_uniform(rand_state) * (depth_max_perturbed - depth_min_perturbed) + depth_min_perturbed;
    } while (depth_perturbed < params.depth_min && depth_perturbed > params.depth_max);

    const int num_planes = 2;  // Reduced from 5 to 2 as we're only testing current and perturbed depth
    float depths[num_planes] = {*depth, depth_perturbed};
    float4 normal = make_float4(plane_hypothesis->x, plane_hypothesis->y, plane_hypothesis->z, 0);  // Keep the normal fixed

    for (int i = 0; i < num_planes; ++i) {
        float cost_vector[32] = {2.0f};
        float4 temp_plane_hypothesis = normal;
        temp_plane_hypothesis.w = GetDistance2Origin(cameras[0], p, depths[i], temp_plane_hypothesis);
        ComputeMultiViewCostVector(images, cameras, p, temp_plane_hypothesis, cost_vector, params);

        float temp_cost = 0.0f;
        for (int j = 0; j < params.num_images - 1; ++j) {
            if (view_weights[j] > 0) {
                if (params.geom_consistency) {
                    temp_cost += view_weights[j] * (cost_vector[j] + 0.1f * ComputeGeomConsistencyCost(depth_images[j+1], cameras[0], cameras[j+1], temp_plane_hypothesis, p));
                }
                else {
                    temp_cost += view_weights[j] * cost_vector[j];
                }
            }
        }
        temp_cost /= weight_norm;

        if (temp_cost < *cost) {
            *depth = depths[i];
            plane_hypothesis->w = temp_plane_hypothesis.w;
            *cost = temp_cost;
        }
    }
}

__device__ void CheckerboardPropagation(const cudaTextureObject_t *images, const cudaTextureObject_t *depths, const Camera *cameras, 
                                        float4 *plane_hypotheses, float *costs, curandState *rand_states, unsigned int *selected_views, 
                                        const int2 p, const PatchMatchParams params, const int iter)
{
    int width = cameras[0].width;
    int height = cameras[0].height;
    if (p.x >= width || p.y >= height) {
        return;
    }

    const int center = p.y * width + p.x;
    int left_near = center - 1;
    int left_far = center - 3;
    int right_near = center + 1;
    int right_far = center + 3;
    int up_near = center - width;
    int up_far = center - 3 * width;
    int down_near = center + width;
    int down_far = center + 3 * width;

    // Adaptive Checkerboard Sampling
    float cost_array[8][32] = {2.0f};
    // 0 -- up_near, 1 -- up_far, 2 -- down_near, 3 -- down_far, 4 -- left_near, 5 -- left_far, 6 -- right_near, 7 -- right_far
    bool flag[8] = {false};
    int num_valid_pixels = 0;

    float costMin;
    int costMinPoint;

    // up_far
    if (p.y > 2) {
        flag[1] = true;
        num_valid_pixels++;
        costMin = costs[up_far];
        costMinPoint = up_far;
        for (int i = 1; i < 11; ++i) {
            if (p.y > 2 + 2 * i) {
                int pointTemp = up_far - 2 * i * width;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        up_far = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[up_far], cost_array[1], params);
    }

    // dwon_far
    if (p.y < height - 3) {
        flag[3] = true;
        num_valid_pixels++;
        costMin = costs[down_far];
        costMinPoint = down_far;
        for (int i = 1; i < 11; ++i) {
            if (p.y < height - 3 - 2 * i) {
                int pointTemp = down_far + 2 * i * width;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        down_far = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[down_far], cost_array[3], params);
    }

    // left_far
    if (p.x > 2) {
        flag[5] = true;
        num_valid_pixels++;
        costMin = costs[left_far];
        costMinPoint = left_far;
        for (int i = 1; i < 11; ++i) {
            if (p.x > 2 + 2 * i) {
                int pointTemp = left_far - 2 * i;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        left_far = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[left_far], cost_array[5], params);
    }

    // right_far
    if (p.x < width - 3) {
        flag[7] = true;
        num_valid_pixels++;
        costMin = costs[right_far];
        costMinPoint = right_far;
        for (int i = 1; i < 11; ++i) {
            if (p.x < width - 3 - 2 * i) {
                int pointTemp = right_far + 2 * i;
                if (costMin < costs[pointTemp]) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        right_far = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[right_far], cost_array[7], params);
    }

    // up_near
    if (p.y > 0) {
        flag[0] = true;
        num_valid_pixels++;
        costMin = costs[up_near];
        costMinPoint = up_near;
        for (int i = 0; i < 3; ++i) {
            if (p.y > 1 + i && p.x > i) {
                int pointTemp = up_near - (1 + i) * width - i;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
            if (p.y > 1 + i && p.x < width - 1 - i) {
                int pointTemp = up_near - (1 + i) * width + i;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        up_near = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[up_near], cost_array[0], params);
    }

    // down_near
    if (p.y < height - 1) {
        flag[2] = true;
        num_valid_pixels++;
        costMin = costs[down_near];
        costMinPoint = down_near;
        for (int i = 0; i < 3; ++i) {
            if (p.y < height - 2 - i && p.x > i) {
                int pointTemp = down_near + (1 + i) * width - i;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
            if (p.y < height - 2 - i && p.x < width - 1 - i) {
                int pointTemp = down_near + (1 + i) * width + i;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        down_near = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[down_near], cost_array[2], params);
    }

    // left_near
    if (p.x > 0) {
        flag[4] = true;
        num_valid_pixels++;
        costMin = costs[left_near];
        costMinPoint = left_near;
        for (int i = 0; i < 3; ++i) {
            if (p.x > 1 + i && p.y > i) {
                int pointTemp = left_near - (1 + i) - i * width;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
            if (p.x > 1 + i && p.y < height - 1 - i) {
                int pointTemp = left_near - (1 + i) + i * width;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        left_near = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[left_near], cost_array[4], params);
    }

    // right_near
    if (p.x < width - 1) {
        flag[6] = true;
        num_valid_pixels++;
        costMin = costs[right_near];
        costMinPoint = right_near;
        for (int i = 0; i < 3; ++i) {
            if (p.x < width - 2 - i && p.y > i) {
                int pointTemp = right_near + (1 + i) - i * width;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
            if (p.x < width - 2 - i && p.y < height - 1- i) {
                int pointTemp = right_near + (1 + i) + i * width;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        right_near = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[right_near], cost_array[6], params);
    }
    const int positions[8] = {up_near, up_far, down_near, down_far, left_near, left_far, right_near, right_far};

    // Multi-hypothesis Joint View Selection
    float view_weights[32] = {0.0f};
    float view_selection_priors[32] = {0.0f};
    int neighbor_positions[4] = {center - width, center + width, center - 1, center + 1};
    for (int i = 0; i < 4; ++i) {
        if (flag[2 * i]) {
            for (int j = 0; j < params.num_images - 1; ++j) {
                if (isSet(selected_views[neighbor_positions[i]], j) == 1) {
                    view_selection_priors[j] += 0.9f;
                } else {
                    view_selection_priors[j] += 0.1f;
                }
            }
        }
    }

    float sampling_probs[32] = {0.0f};
    float cost_threshold = 0.8 * expf((iter) * (iter) / (-90.0f));
    for (int i = 0; i < params.num_images - 1; i++) {
        float count = 0;
        int count_false = 0;
        float tmpw = 0;
        for (int j = 0; j < 8; j++) {
            if (cost_array[j][i] < cost_threshold) {
                tmpw += expf(cost_array[j][i] * cost_array[j][i] / (-0.18f));
                count++;
            }
            if (cost_array[j][i] > 1.2f) {
                count_false++;
            }
        }
        if (count > 2 && count_false < 3) {
            sampling_probs[i] = tmpw / count;
        }
        else if (count_false < 3) {
            sampling_probs[i] = expf(cost_threshold * cost_threshold / (-0.32f));
        }
        sampling_probs[i] = sampling_probs[i] * view_selection_priors[i];
    }

    TransformPDFToCDF(sampling_probs, params.num_images - 1);
    for (int sample = 0; sample < 15; ++sample) {
        const float rand_prob = curand_uniform(&rand_states[center]) - FLT_EPSILON;

        for (int image_id = 0; image_id < params.num_images - 1; ++image_id) {
            const float prob = sampling_probs[image_id];
            if (prob > rand_prob) {
                view_weights[image_id] += 1.0f;
                break;
            }
        }
    }

    unsigned int temp_selected_views = 0;
    int num_selected_view = 0;
    float weight_norm = 0;
    for (int i = 0; i < params.num_images - 1; ++i) {
        if (view_weights[i] > 0) {
            setBit(temp_selected_views, i);
            weight_norm += view_weights[i];
            num_selected_view++;
        }
    }

    float final_costs[8] = {0.0f};
    for (int i = 0; i < 8; ++i) {
        if (flag[i]) {
            float4 temp_plane_hypothesis = plane_hypotheses[center];
            temp_plane_hypothesis.w = GetDistance2Origin(cameras[0], p, ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[positions[i]], p), temp_plane_hypothesis);
            
            for (int j = 0; j < params.num_images - 1; ++j) {
                if (view_weights[j] > 0) {
                    if (params.geom_consistency) {
                        final_costs[i] += view_weights[j] * (cost_array[i][j] + 0.1f * ComputeGeomConsistencyCost(depths[j+1], cameras[0], cameras[j+1], temp_plane_hypothesis, p));
                    }
                    else {
                        final_costs[i] += view_weights[j] * cost_array[i][j];
                    }
                }
            }
            final_costs[i] /= weight_norm;
        }
        else {
            final_costs[i] = FLT_MAX;
        }
    }

    const int min_cost_idx = FindMinCostIndex(final_costs, 8);

    float cost_vector_now[32] = {2.0f};
    ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[center], cost_vector_now, params);
    float cost_now = 0.0f;
    for (int i = 0; i < params.num_images - 1; ++i) {
        if (params.geom_consistency) {
            cost_now += view_weights[i] * (cost_vector_now[i] + 0.1f * ComputeGeomConsistencyCost(depths[i+1], cameras[0], cameras[i+1], plane_hypotheses[center], p));
        }
        else {
            cost_now += view_weights[i] * cost_vector_now[i];
        }
    }
    cost_now /= weight_norm;
    costs[center] = cost_now;
    float depth_now = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);

    if (flag[min_cost_idx]) {
        float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[positions[min_cost_idx]], p);

        if (depth_before >= params.depth_min && depth_before <= params.depth_max && final_costs[min_cost_idx] < cost_now) {
            depth_now = depth_before;
            plane_hypotheses[center] = plane_hypotheses[positions[min_cost_idx]];
            costs[center] = final_costs[min_cost_idx];
            selected_views[center] = temp_selected_views;
        }
    }

    PlaneHypothesisRefinement(images, depths, cameras, &plane_hypotheses[center], &depth_now, &costs[center], &rand_states[center], view_weights, weight_norm, p, params);
}

__global__ void BlackPixelUpdate(cudaTextureObjects *texture_objects, cudaTextureObjects *texture_depths, 
                                 Camera *cameras, float4 *plane_hypotheses, float *costs, curandState *rand_states, 
                                 unsigned int *selected_views, const PatchMatchParams params, const int iter)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) {
        p.y = p.y * 2;
    } else {
        p.y = p.y * 2 + 1;
    }

    CheckerboardPropagation(texture_objects[0].images, texture_depths[0].images, cameras, plane_hypotheses, costs, rand_states, selected_views, p, params, iter);
}

__global__ void RedPixelUpdate(cudaTextureObjects *texture_objects, cudaTextureObjects *texture_depths, 
                               Camera *cameras, float4 *plane_hypotheses, float *costs, curandState *rand_states, 
                               unsigned int *selected_views, const PatchMatchParams params, const int iter)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) {
        p.y = p.y * 2 + 1;
    } else {
        p.y = p.y * 2;
    }

    CheckerboardPropagation(texture_objects[0].images, texture_depths[0].images, cameras, plane_hypotheses, costs, rand_states, selected_views, p, params, iter);
}

__global__ void GetDepthandNormal(Camera *cameras, float4 *plane_hypotheses, const PatchMatchParams params)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    const int width = cameras[0].width;
    const int height = cameras[0].height;

    if (p.x >= width || p.y >= height) {
        return;
    }

    const int center = p.y * width + p.x;
    plane_hypotheses[center].w = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);
    plane_hypotheses[center] = TransformNormal(cameras[0], plane_hypotheses[center]);
}

__device__ void CheckerboardFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs, const int2 p)
{
    int width = cameras[0].width;
    int height = cameras[0].height;
    if (p.x >= width || p.y >= height) {
        return;
    }

    const int center = p.y * width + p.x;

    float filter[21];
    int index = 0;

    filter[index++] = plane_hypotheses[center].w;

    // Left
    const int left = center - 1;
    const int leftleft = center - 3;

    // Up
    const int up = center - width;
    const int upup = center - 3 * width;

    // Down
    const int down = center + width;
    const int downdown = center + 3 * width;

    // Right
    const int right = center + 1;
    const int rightright = center + 3;

    if (costs[center] < 0.001f) {
        return;
    }

    if (p.y>0) {
        filter[index++] = plane_hypotheses[up].w;
    }
    if (p.y>2) {
        filter[index++] = plane_hypotheses[upup].w;
    }
    if (p.y>4) {
        filter[index++] = plane_hypotheses[upup-width*2].w;
    }
    if (p.y<height-1) {
        filter[index++] = plane_hypotheses[down].w;
    }
    if (p.y<height-3) {
        filter[index++] = plane_hypotheses[downdown].w;
    }
    if (p.y<height-5) {
        filter[index++] = plane_hypotheses[downdown+width*2].w;
    }
    if (p.x>0) {
        filter[index++] = plane_hypotheses[left].w;
    }
    if (p.x>2) {
        filter[index++] = plane_hypotheses[leftleft].w;
    }
    if (p.x>4) {
        filter[index++] = plane_hypotheses[leftleft-2].w;
    }
    if (p.x<width-1) {
        filter[index++] = plane_hypotheses[right].w;
    }
    if (p.x<width-3) {
        filter[index++] = plane_hypotheses[rightright].w;
    }
    if (p.x<width-5) {
        filter[index++] = plane_hypotheses[rightright+2].w;
    }
    if (p.y>0 &&
        p.x<width-2) {
        filter[index++] = plane_hypotheses[up+2].w;
    }
    if (p.y< height-1 &&
        p.x<width-2) {
        filter[index++] = plane_hypotheses[down+2].w;
    }
    if (p.y>0 &&
        p.x>1)
    {
        filter[index++] = plane_hypotheses[up-2].w;
    }
    if (p.y<height-1 &&
        p.x>1) {
        filter[index++] = plane_hypotheses[down-2].w;
    }
    if (p.x>0 &&
        p.y>2)
    {
        filter[index++] = plane_hypotheses[left  - width*2].w;
    }
    if (p.x<width-1 &&
        p.y>2)
    {
        filter[index++] = plane_hypotheses[right - width*2].w;
    }
    if (p.x>0 &&
        p.y<height-2) {
        filter[index++] = plane_hypotheses[left  + width*2].w;
    }
    if (p.x<width-1 &&
        p.y<height-2) {
        filter[index++] = plane_hypotheses[right + width*2].w;
    }

    sort_small(filter,index);
    int median_index = index / 2;
    if (index % 2 == 0) {
        plane_hypotheses[center].w = (filter[median_index-1] + filter[median_index]) / 2;
    } else {
        plane_hypotheses[center].w = filter[median_index];
    }
}

__global__ void BlackPixelFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) {
        p.y = p.y * 2;
    } else {
        p.y = p.y * 2 + 1;
    }

    CheckerboardFilter(cameras, plane_hypotheses, costs, p);
}

__global__ void RedPixelFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) {
        p.y = p.y * 2 + 1;
    } else {
        p.y = p.y * 2;
    }

    CheckerboardFilter(cameras, plane_hypotheses, costs, p);
}


void PatchMatch::RunPatchMatch()
{
    const int width = cameras[0].width;
    const int height = cameras[0].height;

    int BLOCK_W = 32;
    int BLOCK_H = (BLOCK_W / 2);

    dim3 grid_size_randinit;
    grid_size_randinit.x = (width + 16 - 1) / 16;
    grid_size_randinit.y=(height + 16 - 1) / 16;
    grid_size_randinit.z = 1;
    dim3 block_size_randinit;
    block_size_randinit.x = 16;
    block_size_randinit.y = 16;
    block_size_randinit.z = 1;

    dim3 grid_size_checkerboard;
    grid_size_checkerboard.x = (width + BLOCK_W - 1) / BLOCK_W;
    grid_size_checkerboard.y= ( (height / 2) + BLOCK_H - 1) / BLOCK_H;
    grid_size_checkerboard.z = 1;
    dim3 block_size_checkerboard;
    block_size_checkerboard.x = BLOCK_W;
    block_size_checkerboard.y = BLOCK_H;
    block_size_checkerboard.z = 1;

    int max_iterations = params.max_iterations;

    RandomInitialization<<<grid_size_randinit, block_size_randinit>>>(texture_objects_cuda, cameras_cuda, plane_hypotheses_cuda, costs_cuda, rand_states_cuda, selected_views_cuda, params, depths_cuda); 
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
     
    for (int i = 0; i < max_iterations; ++i) {

        BlackPixelUpdate<<<grid_size_checkerboard, block_size_checkerboard>>>(texture_objects_cuda, texture_depths_cuda, cameras_cuda, plane_hypotheses_cuda, costs_cuda, rand_states_cuda, selected_views_cuda, params, i);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        
        RedPixelUpdate<<<grid_size_checkerboard, block_size_checkerboard>>>(texture_objects_cuda, texture_depths_cuda, cameras_cuda, plane_hypotheses_cuda, costs_cuda, rand_states_cuda, selected_views_cuda, params, i);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        // printf("iteration: %d\n", i);
    }

    GetDepthandNormal<<<grid_size_randinit, block_size_randinit>>>(cameras_cuda, plane_hypotheses_cuda, params);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    BlackPixelFilter<<<grid_size_checkerboard, block_size_checkerboard>>>(cameras_cuda, plane_hypotheses_cuda, costs_cuda);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    RedPixelFilter<<<grid_size_checkerboard, block_size_checkerboard>>>(cameras_cuda, plane_hypotheses_cuda, costs_cuda);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    cudaMemcpy(plane_hypotheses_host, plane_hypotheses_cuda, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
    cudaMemcpy(costs_host, costs_cuda, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

torch::Tensor propagate_cuda(torch::Tensor images, torch::Tensor intrinsics, torch::Tensor poses, 
                                          torch::Tensor depth, torch::Tensor normal, torch::Tensor depth_intervals, int patch_size)
{
    cudaSetDevice(0);

    PatchMatch pm;
    pm.SetPatchSize(patch_size);

    images = images.to(torch::kFloat);
    
    pm.InuputInitialization(images, intrinsics, poses, depth, normal, depth_intervals);

    pm.CudaSpaceInitialization();
    pm.RunPatchMatch();

    const int width = pm.GetReferenceImageWidth();
    const int height = pm.GetReferenceImageHeight();

    torch::Tensor depths = torch::zeros({height, width}, torch::kFloat);
    torch::Tensor normals = torch::zeros({height, width, 3}, torch::kFloat);

    int numPixels = width * height;

    float4* plane_hypotheses = pm.GetPlaneHypotheses();

    torch::Tensor planeHypothesisTensor = torch::from_blob(plane_hypotheses, {numPixels, 4}, torch::kFloat);

    torch::Tensor propagated_depth = planeHypothesisTensor.index({torch::indexing::Slice(), 3}).reshape({height, width}).unsqueeze(0);

    torch::Tensor propagated_normal = planeHypothesisTensor.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}).reshape({height, width, 3}).permute({2, 0, 1});

    torch::Tensor results = torch::cat({propagated_depth, propagated_normal}, 0);

    return results;
}