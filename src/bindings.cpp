#include <torch/extension.h>

torch::Tensor render(torch::Tensor means3d, torch::Tensor quats, torch::Tensor colors, torch::Tensor opacities, torch::Tensor scales, torch::Tensor viewmat, torch::Tensor K, int width, int height);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("render", &render, "A renderer");
}