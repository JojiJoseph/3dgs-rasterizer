from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='jsplat',
    ext_modules=[
        CUDAExtension(
            name='jsplat',
            sources=['src/bindings.cpp', 'src/rasterizer.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)