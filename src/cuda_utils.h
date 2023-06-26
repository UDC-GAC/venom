
/*
 * Copyright (C) 2023 Roberto Lopez Castro (roberto.lopez.castro@udc.es). All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SPMM_CNN_CUDA_UTILS_H_
#define SPMM_CNN_CUDA_UTILS_H_

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::printf("CUDA API failed at line %d with error: %s (%d)\n",        \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::printf("CUSPARSE API failed at line %d with error: %s (%d)\n",    \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        std::printf("CUBLAS API failed at line %d with error: (%d)\n",         \
               __LINE__, status);                                              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSELT(func)                                                                         \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
    }                                                                          \
}

template<typename F, class... Args>
float cuTime(int times, F&& f, Args&&... args){
  float time_ms;
  cudaEvent_t start, stop;
  int warmup = (times>0)?(10):(0);

  cudaDeviceSynchronize();

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for(int i=0; i<warmup + times; ++i){
    if (i == warmup)
        cudaEventRecord(start, 0);

    f(forward<Args>(args)...);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_ms, start, stop);
  time_ms = time_ms / (float)times;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return time_ms;
}

#endif