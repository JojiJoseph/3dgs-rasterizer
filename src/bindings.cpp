#include <torch/extension.h>

torch::Tensor render(const torch::Tensor& means3d, const torch::Tensor& quats, const torch::Tensor& colors, const torch::Tensor& opacities, const torch::Tensor& scales, const torch::Tensor& viewmat, const torch::Tensor& K, const int width, const int height);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("render", &render, "A renderer");
}