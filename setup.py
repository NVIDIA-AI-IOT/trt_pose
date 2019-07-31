from setuptools import setup, find_packages, Extension
from torch.utils import cpp_extension

# plugins_ext = Extension(
#    name='trt_pose.plugins',
#    sources=['trt_pose/find_peaks.cpp'],
#    include_dirs=cpp_extension.include_paths(),
#    language='c++'
# )

setup(
    name='trt_pose',
    version='0.0.0',
    description='Pose detection accelerated by NVIDIA TensorRT',
    packages=find_packages(),
    ext_package='trt_pose',
    ext_modules=[cpp_extension.CppExtension('plugins', [
        'trt_pose/plugins/find_peaks.cpp',
        'trt_pose/plugins/paf_score_graph.cpp',
        'trt_pose/plugins/refine_peaks.cpp',
        'trt_pose/plugins/plugins.cpp',
        'trt_pose/plugins/munkres.cpp',
#         'trt_pose/plugins/connect_parts.cpp'
    ])],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    install_requires=[
    ],
)