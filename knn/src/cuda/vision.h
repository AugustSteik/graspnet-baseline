#pragma once
#include <torch/extension.h>             // PyTorch C++ API
#include <ATen/cuda/CUDAContext.h>       // CUDA context for streams
#include <c10/cuda/CUDAStream.h>         // CUDA stream utilities
// #include <THC/THC.h>

void knn_device(float* ref_dev, int ref_width,
    float* query_dev, int query_width,
    int height, int k, float* dist_dev, int64_t* ind_dev, cudaStream_t stream);