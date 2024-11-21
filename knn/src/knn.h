#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
// #include <THC/THC.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#endif



int knn(at::Tensor& ref, at::Tensor& query, at::Tensor& idx)
{

    // TODO check dimensions
    int64_t batch, ref_nb, query_nb, dim, k;
    batch = ref.size(0);
    dim = ref.size(1);
    k = idx.size(1);
    ref_nb = ref.size(2);
    query_nb = query.size(2);

    float *ref_dev = ref.data<float>();
    float *query_dev = query.data<float>();
    int64_t *idx_dev = idx.data<int64_t>();




  if (ref.type().is_cuda()) {
#ifdef WITH_CUDA
    // TODO raise error if not compiled with CUDA
    // float *dist_dev = (float*)THCudaMalloc(state, ref_nb * query_nb * sizeof(float));
    
    float *dist_dev;
    cudaError_t err1 = cudaMalloc((void**)&dist_dev, ref_nb * query_nb * sizeof(float));
    if (err1 != cudaSuccess) {
      printf("CUDA malloc failed in knn.h: %s\n", cudaGetErrorString(err1));
      return 0;
    }

    for (int b = 0; b < batch; b++)
    {
    // knn_device(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
    //   dist_dev, idx_dev + b * k * query_nb, THCState_getCurrentStream(state));
      knn_device(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
      dist_dev, idx_dev + b * k * query_nb, c10::cuda::getCurrentCUDAStream());
    }
    // THCudaFree(state, dist_dev);
    cudaFree(dist_dev);
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess)
    {
        printf("error in knn: %s\n", cudaGetErrorString(err2));
        // THError("aborting");
        return 0;
    }
    return 1;
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }


    float *dist_dev = (float*)malloc(ref_nb * query_nb * sizeof(float));
    int64_t *ind_buf = (int64_t*)malloc(ref_nb * sizeof(int64_t));
    for (int b = 0; b < batch; b++) {
    knn_cpu(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
      dist_dev, idx_dev + b * k * query_nb, ind_buf);
    }

    free(dist_dev);
    free(ind_buf);

    return 1;

}
