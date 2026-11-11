#ifndef CAUSA_CUDA_H
#define CAUSA_CUDA_H

#include "causa_tensor.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

causa_backend_t* causa_backend_cuda_new(void);

#ifdef __cplusplus
}
#endif

#endif