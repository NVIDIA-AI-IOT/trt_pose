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
    version='0.0.1',
    description='Pose detection accelerated by NVIDIA TensorRT',
    packages=find_packages(),
    ext_package='trt_pose',
    ext_modules=[cpp_extension.CppExtension('plugins', [
        'trt_pose/parse/find_peaks.cpp',
        'trt_pose/parse/paf_score_graph.cpp',
        'trt_pose/parse/refine_peaks.cpp',
        'trt_pose/parse/munkres.cpp',
        'trt_pose/parse/connect_parts.cpp',
        'trt_pose/plugins.cpp',
        'trt_pose/train/generate_cmap.cpp',
        'trt_pose/train/generate_paf.cpp',
    ])],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    install_requires=[
    ],
)
