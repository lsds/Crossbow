#ifndef __CROSSBOW_DEBUG_H_
#define __CROSSBOW_DEBUG_H_

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include <cudnn.h>

#include <nccl.h>

#undef GPU_VERBOSE
/* #define GPU_VERBOSE */

#undef KERNEL_CHECKSUM
/* #define COMPUTE_CHECKSUM */

#undef  NOOPS
#ifdef  NOOPS
#define KERNEL_NOOP
#define CUBLAS_NOOP
#define CUDANN_NOOP
#define CUDART_NOOP
#define CURAND_NOOP
#endif

#undef dbg
#ifdef GPU_VERBOSE
#	define dbg(fmt, args...) do { fprintf(stdout, "DEBUG %35s (l. %4d) > " fmt, __FILE__, __LINE__, ## args); fflush(stdout); } while (0)
#else
#	define dbg(fmt, args...)
#endif

#define info(fmt, args...) do { fprintf(stdout, "INFO  %35s (l. %4d) > " fmt, __FILE__, __LINE__, ## args); fflush(stdout); } while (0)

#define warn(fmt, args...) do { fprintf(stderr, "WARN  %35s (l. %4d) > " fmt, __FILE__, __LINE__, ## args); fflush(stderr); } while (0)

#define err(fmt, args...) do { fprintf(stderr, "ERROR %35s (l. %4d) > " fmt, __FILE__, __LINE__, ## args); exit(1); } while (0)

/* Exception handling */

/* #undef  CROSSBOW_EXCEPTIONS */
#define CROSSBOW_EXCEPTIONS
#ifdef  CROSSBOW_EXCEPTIONS

#define nullPointerException(x) do { if (x == NULL) { fprintf(stderr, "error: null pointer exception (in %s, %s:%d)\n", __func__, __FILE__, __LINE__); exit (1); } } while (0)

#define indexOutOfBoundsException(x, y) do { if (x < 0 || x > (y - 1)) { fprintf(stderr, "error: array index out of bounds (in %s, %s:%d)\n", __func__, __FILE__, __LINE__); exit (1); } } while (0)

#define invalidArgumentException(cond) do { if (! (cond)) { fprintf(stderr, "error: invalid argument (in %s, %s:%d)\n", __func__, __FILE__, __LINE__); exit (1); } } while (0)

#define illegalOperationException() do { fprintf(stderr, "error: illegal operation (in %s, %s:%d)\n", __func__, __FILE__, __LINE__); exit (1); } while (0)

#define unsupportedOperationException() do { fprintf(stderr, "error: unsupported operation (in %s, %s:%d)\n", __func__, __FILE__, __LINE__); exit (1); } while (0)

#define illegalStateException() do { fprintf(stderr, "error: illegal state (in %s, %s:%d)\n", __func__, __FILE__, __LINE__); exit (1); } while (0)

#define invalidConditionException(cond) do { if (! (cond)) { fprintf(stderr, "error: invalid condition (in %s, %s:%d)\n", __func__, __FILE__, __LINE__); exit (1); } } while (0)

#else /* Ignore all exceptions */

#define nullPointerException(x) ((void) (x))

#define indexOutOfBoundsException(x, y) ((void) (x + y))

#define invalidArgumentException(cond) ((void) (cond))

#define invalidConditionException(cond) ((void) (cond))

#define illegalOperationException() do { fprintf(stderr, "error: illegal operation (in %s, %s:%d)\n", __func__, __FILE__, __LINE__); exit (1); } while (0)

#define unsupportedOperationException() do { fprintf(stderr, "error: unsupported operation (in %s, %s:%d)\n", __func__, __FILE__, __LINE__); exit (1); } while (0)

#define illegalStateException() do { fprintf(stderr, "error: illegal state (in %s, %s:%d)\n", __func__, __FILE__, __LINE__); exit (1); } while (0)

#endif

/* CUDA error handling */

static const char *mycudaGetErrorEnum(cublasStatus_t error) {
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

       case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
                    return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
        	return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}

#define checkCudaErrors(x) do { if (x != cudaSuccess) { fprintf(stderr, "cuda error: %s (in %s, %s:%d)\n", cudaGetErrorString(x), __func__, __FILE__, __LINE__); exit (1); } } while (0)

#define checkCublasStatus(x) do { if (x != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublas error: %s (in %s, %s:%d)\n", mycudaGetErrorEnum(x), __func__, __FILE__, __LINE__); exit (1); } } while (0)

#define checkCudnnStatus(x) do { if (x != CUDNN_STATUS_SUCCESS) { fprintf(stderr, "cudnn error: %s (in %s, %s:%d)\n", cudnnGetErrorString(x), __func__, __FILE__, __LINE__); exit (1); } } while (0)

#define checkCurandStatus(x) do { if (x != CURAND_STATUS_SUCCESS) { fprintf(stderr, "curand error: %d (in %s, %s:%d)\n", x, __func__, __FILE__, __LINE__); exit (1); } } while (0)

#define checkNcclErrors(x) do { if (x != ncclSuccess) { fprintf(stderr, "nccl error: %s (in %s, %s:%d)\n", ncclGetErrorString(x), __func__, __FILE__, __LINE__); exit (1); } } while (0)

#endif /* __GPU_DEBUG_H_ */
