/*! \file reduce_kernels.h
 *  \brief Contains the OpenCL and CUDA kernels for the Reduce skeleton (used for both 1D and 2D reduce operation).
 */

#ifndef REDUCE_KERNELS_H
#define REDUCE_KERNELS_H


namespace skepu2
{
namespace backend
{

/*!
 *  \ingroup kernels
 */

/*!
 *  \defgroup ReduceKernels Reduce Kernels
 *
 *  Definitions of CUDA and OpenCL kernels for the Reduce skeleton.
 * \{
 */



/*!
 * \brief A small helper to determine whether the number is a power of 2.
 *
 * \param x the actual number.
 * \return bool specifying whether number of power of 2 or not,
 */
inline bool isPow2(size_t x)
{
   return ((x&(x-1))==0);
}


/*!
 * \brief A helper to return a value that is nearest value that is power of 2.
 *
 * \param x The input number for which we need to find the nearest value that is power of 2.
 * \return The nearest value that is power of 2.
 */
inline size_t nextPow2(size_t x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}




/*!
 * Compute the number of threads and blocks to use for the reduction kernel.
 * We set threads / block to the minimum of maxThreads and n/2 where n is
 * problem size. We observe the maximum specified number of blocks, because
 * each kernel thread can process more than 1 elements.
 *
 * \param n Problem size.
 * \param maxBlocks Maximum number of blocks that can be used.
 * \param maxThreads Maximum number of threads that can be used.
 */
inline std::pair<size_t, size_t> getNumBlocksAndThreads(size_t n, size_t maxBlocks, size_t maxThreads)
{
	const size_t threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	const size_t blocks = std::min(maxBlocks, (n + (threads * 2 - 1)) / (threads * 2));
	return std::make_pair(threads, blocks);
}



#ifdef SKEPU_CUDA

/*!
 * Helper method used to call the actual CUDA kernel for reduction. Used when PINNED MEMORY is disabled
 *
 *  \param reduceFunc The reduction user function to be used.
 *  \param size size of the input array to be reduced.
 *  \param numThreads Number of threads to be used for kernel execution.
 *  \param numBlocks Number of blocks to be used for kernel execution.
 *  \param d_idata CUDA memory pointer to input array.
 *  \param d_odata CUDA memory pointer to output array.
 *  \param enableIsPow2 boolean flag (default true) used to enable/disable isPow2 optimizations. disabled only for sparse row-/column-wise reduction for technical reasons.
 */
template <typename ReduceFunc, typename T>
void CallReduceKernel(ReduceFunc kernel, size_t size, size_t numThreads, size_t numBlocks, T *d_idata, T *d_odata, bool enableIsPow2=true)
{
	dim3 dimBlock(numThreads, 1, 1);
	dim3 dimGrid(numBlocks, 1, 1);
	
	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	size_t smemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T);
	
	// choose which of the optimized versions of reduction to launch
	if (isPow2(size) && enableIsPow2)
	{
		switch (numThreads)
		{
		case 512: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 512, true); break;
		case 256: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 256, true); break;
		case 128: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 128, true); break;
		case  64: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,  64, true); break;
		case  32: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,  32, true); break;
		case  16: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,  16, true); break;
		case   8: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,   8, true); break;
		case   4: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,   4, true); break;
		case   2: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,   2, true); break;
		case   1: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,   1, true); break;
		}
	}
	else
	{
		switch (numThreads)
		{
		case 512: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 512, false); break;
		case 256: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 256, false); break;
		case 128: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, 128, false); break;
		case  64: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,  64, false); break;
		case  32: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,  32, false); break;
		case  16: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,  16, false); break;
		case   8: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,   8, false); break;
		case   4: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,   4, false); break;
		case   2: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,   2, false); break;
		case   1: kernel<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size,   1, false); break;
		}
	}
}




#ifdef USE_PINNED_MEMORY
/*!
 * Helper method used to call the actual CUDA kernel for reduction. Used when PINNED MEMORY is enabled
 *
 *  \param reduceFunc The reduction user function to be used.
 *  \param size size of the input array to be reduced.
 *  \param numThreads Number of threads to be used for kernel execution.
 *  \param numBlocks Number of blocks to be used for kernel execution.
 *  \param d_idata CUDA memory pointer to input array.
 *  \param d_odata CUDA memory pointer to output array.
 *  \param stream CUDA stream to be used.
 *  \param enableIsPow2 boolean flag (default true) used to enable/disable isPow2 optimizations. disabled only for sparse row-/column-wise reduction for technical reasons.
 */
template<typename ReduceFunc, typename T>
void CallReduceKernel_WithStream(ReduceFunc kernel, size_t size, size_t numThreads, size_t numBlocks, T *d_idata, T *d_odata, cudaStream_t &stream, bool enableIsPow2=true)
{
	dim3 dimBlock(numThreads, 1, 1);
	dim3 dimGrid(numBlocks, 1, 1);
	
	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	size_t smemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T);
	
	// choose which of the optimized versions of reduction to launch
	if (isPow2(size) && enableIsPow2)
	{
		switch (numThreads)
		{
		case 512: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size, 512, true); break;
		case 256: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size, 256, true); break;
		case 128: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size, 128, true); break;
		case  64: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,  64, true); break;
		case  32: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,  32, true); break;
		case  16: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,  16, true); break;
		case   8: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,   8, true); break;
		case   4: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,   4, true); break;
		case   2: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,   2, true); break;
		case   1: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,   1, true); break;
		}
	}
	else
	{
		switch (numThreads)
		{
		case 512: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size, 512, false); break;
		case 256: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size, 256, false); break;
		case 128: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size, 128, false); break;
		case  64: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,  64, false); break;
		case  32: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,  32, false); break;
		case  16: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,  16, false); break;
		case   8: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,   8, false); break;
		case   4: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,   4, false); break;
		case   2: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,   2, false); break;
		case   1: kernel<<< dimGrid, dimBlock, smemSize, stream >>>(d_idata, d_odata, size,   1, false); break;
		}
	}
}
#endif



#ifdef USE_PINNED_MEMORY
/*!
 *  A helper function that is used to call the actual kernel for reduction. Used by other functions to call the actual kernel
 *  Internally, it just calls 2 kernels by setting their arguments. No synchronization is enforced.
 *
 *  \param reduceFunc The reduction user function to be used.
 *  \param n size of the input array to be reduced.
 *  \param numThreads Number of threads to be used for kernel execution.
 *  \param numBlocks Number of blocks to be used for kernel execution.
 *  \param maxThreads Maximum number of threads that can be used for kernel execution.
 *  \param maxBlocks Maximum number of blocks that can be used for kernel execution.
 *  \param d_idata CUDA memory pointer to input array.
 *  \param d_odata CUDA memory pointer to output array.
 *  \param deviceID Integer deciding which device to utilize.
 *  \param stream CUDA stream to be used, only when using Pinned memory allocations.
 *  \param enableIsPow2 boolean flag (default true) used to enable/disable isPow2 optimizations. disabled only for sparse row-/column-wise reduction for technical reasons.
 */
template <typename ReduceFunc, typename T>
void ExecuteReduceOnADevice(ReduceFunc kernel, size_t n, size_t  numThreads, size_t  numBlocks, size_t  maxThreads, size_t  maxBlocks, T* d_idata, T* d_odata, unsigned int deviceID, cudaStream_t &stream, bool enableIsPow2 = true)
#else
/*!
 *  A helper function that is used to call the actual kernel for reduction. Used by other functions to call the actual kernel
 *  Internally, it just calls 2 kernels by setting their arguments. No synchronization is enforced.
 *
 *  \param reduceFunc The reduction user function to be used.
 *  \param n size of the input array to be reduced.
 *  \param numThreads Number of threads to be used for kernel execution.
 *  \param numBlocks Number of blocks to be used for kernel execution.
 *  \param maxThreads Maximum number of threads that can be used for kernel execution.
 *  \param maxBlocks Maximum number of blocks that can be used for kernel execution.
 *  \param d_idata CUDA memory pointer to input array.
 *  \param d_odata CUDA memory pointer to output array.
 *  \param deviceID Integer deciding which device to utilize.
 *  \param enableIsPow2 boolean flag (default true) used to enable/disable isPow2 optimizations. disabled only for sparse row-/column-wise reduction for technical reasons.
 */
template <typename T, typename ReduceFunc>
void ExecuteReduceOnADevice(ReduceFunc kernel, size_t  n, size_t  numThreads, size_t  numBlocks, size_t  maxThreads, size_t  maxBlocks, T* d_idata, T* d_odata, unsigned int deviceID, bool enableIsPow2 = true)
#endif
{
	// execute the kernel
#ifdef USE_PINNED_MEMORY
	CallReduceKernel_WithStream(kernel, n, numThreads, numBlocks, d_idata, d_odata, stream, enableIsPow2);
#else
	CallReduceKernel(kernel, n, numThreads, numBlocks, d_idata, d_odata, enableIsPow2);
#endif
	
	// sum partial block sums on GPU
	size_t s = numBlocks;
	
	while(s > 1)
	{
		size_t blocks, threads;
		std::tie(threads, blocks) = getNumBlocksAndThreads(s, maxBlocks, maxThreads);
		
#ifdef USE_PINNED_MEMORY
		CallReduceKernel_WithStream(kernel, s, threads, blocks, d_odata, d_odata, stream, enableIsPow2);
#else
		CallReduceKernel(kernel, s, threads, blocks, d_odata, d_odata, enableIsPow2);
#endif
		
		s = (s + (threads*2-1)) / (threads*2);
	}
}

#endif

/*!
 *  \}
 */

}
}


#endif

