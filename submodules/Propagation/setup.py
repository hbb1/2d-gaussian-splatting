from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='gaussianpro',
    ext_modules=[
        CUDAExtension('gaussianpro',
            sources=[
                'PatchMatch.cpp',
                'Propagation.cu',
                'pro.cpp'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-gencode=arch=compute_80,code=sm_80',
                ]
            },
            ),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)