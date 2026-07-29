from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, ROCM_HOME


if torch.version.hip is None or ROCM_HOME is None:
    raise RuntimeError("gfx1010-kernels native extension requires ROCm PyTorch")


setup(
    ext_modules=[
        CUDAExtension(
            "gfx1010_kernels._C",
            sources=[
                "csrc/residual_layer_norm_bindings.cpp",
                "csrc/residual_layer_norm_kernel.hip",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--offload-arch=gfx1010",
                    "-fno-gpu-rdc",
                ],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=True,
            no_python_abi_suffix=False,
        )
    },
)
