#ifndef __CROSSBOW_CUDA_MACROS_H_
#define __CROSSBOW_CUDA_MACROS_H_

#define CUDA_KERNEL_LOOP(ndx, max) \
	for (int ndx = ((blockIdx.x * blockDim.x) + threadIdx.x); ndx < (max); ndx += (blockDim.x * gridDim.x))

#if __CUDA_ARCH__ >= 200
	const int CUDA_NUM_THREADS = 1024;
#else
	const int CUDA_NUM_THREADS = 512;
#endif

inline int GET_BLOCKS (const int n) { return ((n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS); }

inline void ignore(void * __attribute__((unused)) parameter) { (void) parameter; }

#define UNUSED(x) ignore(&x)

#endif /* __CROSSBOW_CUDA_MACROS_H_ */
