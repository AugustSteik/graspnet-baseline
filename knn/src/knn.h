#pragma once
#include "cpu/vision.h"
// #include <ATen/cuda/ThrustAllocator.h>
// #include <ATen/ceil_div.h>

#ifdef WITH_CUDA
#include "cuda/vision.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
// #include <THC/THC.h>
// extern THCState *state;
#endif



int knn(at::Tensor& ref, at::Tensor& query, at::Tensor& idx)
{

    // TODO check dimensions
    long batch, ref_nb, query_nb, dim, k;
    batch = ref.size(0);
    dim = ref.size(1);
    k = idx.size(1);
    ref_nb = ref.size(2);
    query_nb = query.size(2);

    float *ref_dev = ref.data_ptr<float>();
    float *query_dev = query.data_ptr<float>();
    int64_t *idx_dev = idx.data_ptr<int64_t>();




  if (ref.device().is_cuda()) {
#ifdef WITH_CUDA
    // TODO raise error if not compiled with CUDA
    // float *dist_dev = (float*)THCudaMalloc(state, ref_nb * query_nb * sizeof(float));

    // TODO
    // float *dist_dev = (float*)c10::cuda::CUDACachingAllocator::raw_alloc(ref_nb * query_nb * sizeof(float));
    // float *dist_dev = (float)torch::cuda::storage::Allocator::get().raw_allocate(ref_nb * query_nb * sizeof(float)); //GPT
    float *dist_dev;
    cudaMalloc(&dist_dev, ref_nb * query_nb * sizeof(float));

    for (int b = 0; b < batch; b++)
    {
    // knn_device(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
    //   dist_dev, idx_dev + b * k * query_nb, THCState_getCurrentStream(state));
      knn_device(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
      dist_dev, (long*)idx_dev + b * k * query_nb, c10::cuda::getCurrentCUDAStream());
    }
    // THCudaFree(state, dist_dev);

    // TODO
    // c10::cuda::CUDACachingAllocator::raw_delete(dist_dev);
    cudaFree(dist_dev);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("error in knn: %s\n", cudaGetErrorString(err));
        // THError("aborting");
        return 0;
    }
    return 1;
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }


    float *dist_dev = (float*)malloc(ref_nb * query_nb * sizeof(float));
    long *ind_buf = (long*)malloc(ref_nb * sizeof(long));
    for (int b = 0; b < batch; b++) {
    knn_cpu(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
      dist_dev, (long*)idx_dev + b * k * query_nb, ind_buf);
    }

    free(dist_dev);
    free(ind_buf);

    return 1;

}
