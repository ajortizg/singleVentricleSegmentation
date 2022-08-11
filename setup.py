import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, library_paths

if sys.platform=='win32':
    library_dirs = []
else:
    library_dirs = [v +"/" for v in library_paths()]
    
# setup(
#     name='tvl1OF3d',
#     ext_package='tvl1OF3d_cuda_ext',
#     ext_modules=[
#         CUDAExtension('tvl1OF3d',
#             sources=['src/tvl1OF3d.cpp', 'src/tvl1OF3d_kernel.cu'],
#             runtime_library_dirs = library_dirs,
#             extra_compile_args={'cxx': [], 'nvcc': ['-O3']}),
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     })

setup(
    name='opticalFlow',
    ext_package='opticalFlow_cuda_ext',
    ext_modules=[
        CUDAExtension('opticalFlow',
            sources=['src/bindings.cpp', 'src/differentialOps.cu', 'src/anisotropicDifferentialOps.cu', 'src/rotationOps.cu', 'src/warpingOps.cu', 'src/warpingOpsCNN.cu', 'src/prolongationOps.cu', 'src/opticalFlowOps.cu', 'src/ROFOps.cu'],
            runtime_library_dirs = library_dirs,
            extra_compile_args={'cxx': [], 'nvcc': ['-O3']}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })