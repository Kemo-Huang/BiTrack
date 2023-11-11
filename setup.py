import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name="%s.%s" % (module, name),
        sources=[os.path.join(*module.split("."), src) for src in sources],
    )
    return cuda_ext


if __name__ == "__main__":
    setup(
        name="BiTrack",
        cmdclass={
            "build_ext": BuildExtension,
        },
        ext_modules=[
            make_cuda_ext(
                name="iou3d_nms_cuda",
                module="detection.voxel_rcnn.iou3d_nms",
                sources=[
                    "src/iou3d_nms_api.cpp",
                    "src/iou3d_nms.cpp",
                    "src/iou3d_nms_kernel.cu",
                ],
            ),
            make_cuda_ext(
                name="pointnet2_stack_cuda",
                module="detection.voxel_rcnn.pointnet2_stack",
                sources=[
                    "src/pointnet2_api.cpp",
                    "src/ball_query.cpp",
                    "src/ball_query_gpu.cu",
                    "src/group_points.cpp",
                    "src/group_points_gpu.cu",
                    "src/sampling.cpp",
                    "src/sampling_gpu.cu",
                    "src/interpolate.cpp",
                    "src/interpolate_gpu.cu",
                    "src/voxel_query.cpp",
                    "src/voxel_query_gpu.cu",
                    "src/vector_pool.cpp",
                    "src/vector_pool_gpu.cu",
                ],
            ),
        ],
    )
