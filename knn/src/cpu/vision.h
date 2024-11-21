#pragma once
#include <torch/extension.h>

void knn_cpu(float* ref_dev, int ref_width,
    float* query_dev, int query_width,
    int height, int k, float* dist_dev, int64_t* ind_dev, int64_t* ind_buf);