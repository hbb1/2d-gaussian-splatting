from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='gaussianpro',
    ext_modules=[
        CUDAExtension('gaussianpro',
            include_dirs=['/data/chenyt/Code/opencv346/include', '/usr/local/cuda-11.8/include', '.'],
            library_dirs=['/data/chenyt/Code/opencv346/lib/', '/data/chenyt/.conda/2dgs/lib'],
            runtime_library_dirs=['/data/chenyt/Code/opencv346/lib/'],
            libraries=['opencv_core', 'opencv_imgproc', 'opencv_highgui', 'opencv_imgcodecs'],  
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
