#include <torch/extension.h>
#include <vector>

torch::Tensor propagate_cuda(
    torch::Tensor images,
    torch::Tensor intrinsics,
    torch::Tensor poses,
    torch::Tensor depth,
    torch::Tensor normal,
    torch::Tensor depth_intervals,
    int patch_size);

torch::Tensor propagate(
    torch::Tensor images,
    torch::Tensor intrinsics,
    torch::Tensor poses,
    torch::Tensor depth,
    torch::Tensor normal,
    torch::Tensor depth_intervals,
    int patch_size) {

  return propagate_cuda(images, intrinsics, poses, depth, normal, depth_intervals, patch_size);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // bundle adjustment kernels
  m.def("propagate", &propagate, "plane propagation");
}