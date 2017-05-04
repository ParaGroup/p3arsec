/*! \file reduce_cu.inl
 *  \brief Contains the definitions of CUDA specific member functions for the Reduce skeleton.
 */

#ifdef SKEPU_CUDA

#include <cuda.h>

#include "../../reduce_helpers.h"

namespace skepu2
{
	namespace backend
	{
		/*!
		 *  Performs the Reduction on a Matrix with \em CUDA as backend either row-wise or column-wise. Returns a scalar result. The function
		 *  uses only \em one device which is decided by a parameter. A Helper method.
		 */
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		Vector<typename ReduceFunc::Ret> &Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::reduceSingleThreadOneDim_CU(size_t deviceID, Vector<T> &res, Matrix<T> &arg)
		{
			cudaSetDevice(deviceID);
			Device_CU *device = this->m_environment->m_devices_CU[deviceID];
			unsigned int maxKernelsSupported = device->getNoConcurrentKernels();
			const size_t rows = arg.total_rows();
			const size_t cols = arg.total_cols();
			const size_t size = rows * cols;
			const size_t maxBlocks = this->m_selected_spec->GPUBlocks();
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			
			// Setup parameters
			size_t numBlocks, numThreads;
			std::tie(numThreads, numBlocks) = getNumBlocksAndThreads(cols, maxBlocks, maxThreads);
			
			// Copies "all" elements to the device at once, better?
			typename Matrix<T>::device_pointer_type_cu in_mem_p = arg.updateDevice_CU(arg.getAddress(), size, deviceID, AccessMode::Read);
			
			// Manually allocate output memory in this case,
			T *deviceMemPointer;
			allocateCudaMemory<T>(&deviceMemPointer, rows*numBlocks);
			
			T *d_input = in_mem_p->getDeviceDataPointer();
			T *d_output = deviceMemPointer;
			
			// First reduce all elements row-wise so that each row produces one element.
			for (size_t r = 0; r<rows; r++)
			{
#ifdef USE_PINNED_MEMORY
				ExecuteReduceOnADevice<T>(this->m_cuda_kernel, cols, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, deviceID, device->m_streams[r % maxKernelsSupported]);
#else
				ExecuteReduceOnADevice<T>(this->m_cuda_kernel, cols, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, deviceID);
#endif

#ifdef USE_PINNED_MEMORY
				copyDeviceToHost(&res[r], d_output, 1, device->m_streams[r % maxKernelsSupported]);
#else
				copyDeviceToHost(&res[r], d_output, 1);
#endif
				
				d_input += cols;
				d_output += numBlocks;
			}
			
			cutilDeviceSynchronize(); //Do CUTIL way, more safe approach,
			freeCudaMemory<T>(deviceMemPointer);
			
			return res;
		}
		
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		Vector<typename ReduceFunc::Ret> &Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::reduceMultipleOneDim_CU(size_t numDevices, Vector<T> &res, Matrix<T> &arg)
		{
			const size_t rows = arg.total_rows();
			const size_t cols = arg.total_cols();
			const size_t numRowsPerSlice = rows / numDevices;
			const size_t restRows = rows % numDevices;
			const size_t maxBlocks = this->m_selected_spec->GPUBlocks();
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			
			// Setup parameters
			size_t numThreads, numBlocks;
			std::tie(numThreads, numBlocks) = getNumBlocksAndThreads(cols, maxBlocks, maxThreads);
			
			typename Matrix<T>::device_pointer_type_cu in_mem_p[MAX_GPU_DEVICES];
			
			T *deviceMemPointers[MAX_GPU_DEVICES];
			
			// First create CUDA memory if not created already.
			for (size_t i = 0; i < numDevices; ++i)
			{
				cudaSetDevice(i);
				const size_t numRows = numRowsPerSlice + ((i == numDevices - 1) ? restRows : 0);

				in_mem_p[i] = arg.updateDevice_CU((arg.getAddress() + i * numRowsPerSlice * cols), numRows * cols, i, AccessMode::None);
				
				size_t outSize = numRows * numBlocks;
				
				// Manually allocate output memory in this case
				allocateCudaMemory<T>(&deviceMemPointers[i], outSize);
			}
			
			T *d_input = NULL, *d_output = NULL;
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				cudaSetDevice(i);
				Device_CU *device = m_environment->m_devices_CU[i];
				const size_t maxKernelsSupported = device->getNoConcurrentKernels();
				const size_t numRows = numRowsPerSlice + ((i == numDevices-1) ? restRows : 0);
				
				in_mem_p[i] = arg.updateDevice_CU((arg.getAddress() + i * numRowsPerSlice * cols), numRows * cols, i, AccessMode::Read);
				
				d_input = in_mem_p[i]->getDeviceDataPointer();
				d_output = deviceMemPointers[i];
				
				// Reduce all elements row-wise so that each row produces one element.
				for (size_t r = 0; r < numRows; r++)
				{
#ifdef USE_PINNED_MEMORY
					ExecuteReduceOnADevice(this->m_cuda_kernel, cols, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, i, device->m_streams[r % maxKernelsSupported]);
#else
					ExecuteReduceOnADevice(this->m_cuda_kernel, cols, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, i);
#endif

#ifdef USE_PINNED_MEMORY
					copyDeviceToHost(&res[r + numRowsPerSlice * i], d_output, 1, device->m_streams[r % maxKernelsSupported]);
#else
					copyDeviceToHost(&res[r + numRowsPerSlice * i], d_output, 1);
#endif
					
					if (r != numRows - 1)
					{
						d_input += cols;
						d_output += numBlocks;
					}
				}
			}
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				cudaSetDevice(i);
				cutilDeviceSynchronize();
				
				freeCudaMemory<T>(deviceMemPointers[i]);
			}
			
			return res;
		}
		
		
		/*!
		 *  Performs the Reduction on a whole Matrix either row-wise or column-wise. Returns a \em SkePU vector of reduction result.
		 *  Using \em CUDA as backend.
		 */
		template <typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		Vector<typename ReduceFunc::Ret> &Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::CU(Vector<T> &res, Matrix<T>& arg)
		{
			DEBUG_TEXT_LEVEL1("CUDA Reduce (Matrix 1D): rows = " << arg.total_rows() << ", cols = " << arg.total_cols()
				<< ", maxDevices = " << this->m_selected_spec->devices() << ", maxBlocks = " << this->m_selected_spec->GPUBlocks()
				<< ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CU.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			if (numDevices <= 1)
				return this->reduceSingleThreadOneDim_CU(m_environment->bestCUDADevID, res, arg);
				
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
				
			return this->reduceMultipleOneDim_CU(numDevices, res, arg);
		}
		
		
		/*!
		 *  Performs the Reduction on a range of elements with \em CUDA as backend. Returns a scalar result. The function
		 *  uses only \em one device which is decided by a parameter. A Helper method.
		 */
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		template<typename Iterator>
		typename ReduceFunc::Ret Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::reduceSingleThread_CU(size_t deviceID, size_t size, T &res, Iterator arg)
		{
			cudaSetDevice(deviceID);
			Device_CU *device = this->m_environment->m_devices_CU[deviceID];
			const size_t maxBlocks = this->m_selected_spec->GPUBlocks();
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			size_t numBlocks, numThreads;
			std::tie(numThreads, numBlocks) = getNumBlocksAndThreads(size, maxBlocks, maxThreads);
			
			// Copies "all" elements to the device at once, better?
			typename Iterator::device_pointer_type_cu in_mem_p = arg.getParent().updateDevice_CU(arg.getAddress(), size, deviceID, AccessMode::Read);
			DeviceMemPointer_CU<T> out_mem_p(&res, numBlocks, device);
			
#ifdef USE_PINNED_MEMORY
			ExecuteReduceOnADevice(this->m_cuda_kernel, size, numThreads, numBlocks, maxThreads, maxBlocks, in_mem_p->getDeviceDataPointer(), out_mem_p.getDeviceDataPointer(), deviceID, device->m_streams[0]);
#else
			ExecuteReduceOnADevice(this->m_cuda_kernel, size, numThreads, numBlocks, maxThreads, maxBlocks, in_mem_p->getDeviceDataPointer(), out_mem_p.getDeviceDataPointer(), deviceID);
#endif
			out_mem_p.changeDeviceData();
			out_mem_p.copyDeviceToHost(1);
			
#ifdef USE_PINNED_MEMORY // ensure synchronization...
			cutilDeviceSynchronize(); //Do CUTIL way, more safe approach, if result is incorrect could move it up.....
#endif
			
			return res;
		}
		
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		template<typename Iterator>
		typename ReduceFunc::Ret Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::reduceMultiple_CU(size_t numDevices, size_t size, T &res, Iterator arg)
		{
			const size_t numElemPerSlice = size / numDevices;
			const size_t rest = size % numDevices;
			const size_t maxBlocks = this->m_selected_spec->GPUBlocks();
			const size_t maxThreads = this->m_selected_spec->GPUThreads();

			T result[MAX_GPU_DEVICES];
			typename Iterator::device_pointer_type_cu in_mem_p[MAX_GPU_DEVICES];
			typename Iterator::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];
			
			size_t numThreads[MAX_GPU_DEVICES];
			size_t numBlocks[MAX_GPU_DEVICES];
			
			// First create CUDA memory if not created already.
			for (size_t i = 0; i < numDevices; ++i)
			{
				cudaSetDevice(i);
				const size_t numElem = numElemPerSlice + ((i == numDevices - 1) ? rest : 0);
				std::tie(numThreads[i], numBlocks[i]) = getNumBlocksAndThreads(numElem, maxBlocks, maxThreads);
				
				in_mem_p[i] = arg.getParent().updateDevice_CU((arg + i * numElemPerSlice).getAddress(), numElem, i, AccessMode::None);
				
				// Create the output memory
				out_mem_p[i] = new DeviceMemPointer_CU<T>(&result[i], numBlocks[i], this->m_environment->m_devices_CU[i]);
			}
			
			// Create argument structs for all threads
			// Fill out argument struct with right information and start threads.
			for (size_t i = 0; i < numDevices; ++i)
			{
				cudaSetDevice(i);
				const size_t numElem = numElemPerSlice + ((i == numDevices-1) ? rest : 0);
				
				in_mem_p[i] = arg.getParent().updateDevice_CU((arg + i * numElemPerSlice).getAddress(), numElem, i, AccessMode::Read);
				
#ifdef USE_PINNED_MEMORY
				ExecuteReduceOnADevice<T>(this->m_cuda_kernel, numElem, numThreads[i], numBlocks[i], maxThreads, maxBlocks, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), i, this->m_environment->m_devices_CU[i]->m_streams[0]);
#else
				ExecuteReduceOnADevice<T>(this->m_cuda_kernel, numElem, numThreads[i], numBlocks[i], maxThreads, maxBlocks, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), i);
#endif
				//Just mark data change
				out_mem_p[i]->changeDeviceData();
				
#ifdef USE_PINNED_MEMORY // then make copy as it is asynchronous
				out_mem_p[i]->copyDeviceToHost(1);
#endif
			}
			
			// Joins the threads and reduces the results on the CPU, yielding the total result.
			cudaSetDevice(0);
			
#ifdef USE_PINNED_MEMORY
			// if pinned, just synchornize
			cutilDeviceSynchronize();
#else
			// in normal case copy here...
			out_mem_p[0]->copyDeviceToHost(1);
#endif
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				cudaSetDevice(i);
				
#ifdef USE_PINNED_MEMORY
				// if pinned, just synchornize
				cutilDeviceSynchronize();
#else
				// in normal case copy here...
				out_mem_p[i]->copyDeviceToHost(1);
#endif
				res = ReduceFunc::CPU(res, result[i]);
				delete out_mem_p[i];
			}
			
			cudaSetDevice(m_environment->bestCUDADevID);
		
			return res;
		}
		
		
		/*!
		 *  Performs the Reduction on a range of elements. Returns a scalar result. The function decides whether to perform
		 *  the reduction on one device, calling reduceSingleThread_CU(InputIterator inputBegin, InputIterator inputEnd, int deviceID) or
		 *  on multiple devices, dividing the range of elements equally among the participating devices each reducing
		 *  its part. The results are then reduced themselves on the CPU.
		 */
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		template<typename Iterator>
		typename ReduceFunc::Ret Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::CU(size_t size, T &res, Iterator arg)
		{
			DEBUG_TEXT_LEVEL1("CUDA Reduce: size = " << size << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CU.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			if (numDevices <= 1)
				return this->reduceSingleThread_CU(m_environment->bestCUDADevID, size, res, arg);
			
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			return this->reduceMultiple_CU(numDevices, size, res, arg);
		}
		
		
		/*!
		 * Performs the 2D Reduction (First row-wise then column-wise) on an input
		 *  matrix by using \em CUDA backend. Returns a scalar result. The function
		 *  uses only \em one CUDA device which is decided by a parameter.
		 */
		template<typename ReduceFuncRowWise, typename ReduceFuncColWise, typename CUDARowWise, typename CUDAColWise, typename CLKernel>
		typename ReduceFuncRowWise::Ret Reduce2D<ReduceFuncRowWise, ReduceFuncColWise, CUDARowWise, CUDAColWise, CLKernel>
		::reduceSingleThread_CU(size_t deviceID, T &res, Matrix<T>& arg)
		{
			cudaSetDevice(deviceID);
			Device_CU *device = this->m_environment->m_devices_CU[deviceID];
			const size_t maxKernelsSupported = device->getNoConcurrentKernels();
			const size_t maxBlocks = this->m_selected_spec->GPUBlocks();
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			
			// Setup parameters
			const size_t rows = arg.total_rows();
			const size_t cols = arg.total_cols();
			const size_t size = rows * cols;
			
			size_t numThreads, numBlocks;
			std::tie(numThreads, numBlocks) = getNumBlocksAndThreads(cols, maxBlocks, maxThreads);
			
			std::vector<T, malloc_allocator<T> > tempResult(rows);
			
			// Copies "all" elements to the device at once, better?
			typename Matrix<T>::device_pointer_type_cu in_mem_p = arg.updateDevice_CU(arg.getAddress(), size, deviceID, AccessMode::Read);
			
			cutilSafeCall(cudaStreamSynchronize(device->m_streams[0]));
			
			// Manually allocate output memory in this case, if only 1 block allocate for two
			T *deviceMemPointer;
			allocateCudaMemory<T>(&deviceMemPointer, rows * ((numBlocks > 1) ? numBlocks : 2));
			
			T *d_input = in_mem_p->getDeviceDataPointer();
			T *d_output = deviceMemPointer;
			
			// First reduce all elements row-wise so that each row produces one element.
			for (size_t r = 0; r < rows; r++)
			{
#ifdef USE_PINNED_MEMORY
				ExecuteReduceOnADevice(this->m_cuda_kernel, cols, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, deviceID, device->m_streams[r % maxKernelsSupported]);
#else
				ExecuteReduceOnADevice(this->m_cuda_kernel, cols, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, deviceID);
#endif

#ifdef USE_PINNED_MEMORY
				copyDeviceToHost(&tempResult[r], d_output, 1, device->m_streams[r % maxKernelsSupported]);
#else
				copyDeviceToHost(&tempResult[r], d_output, 1);
#endif
				
				d_input += cols;
				d_output += numBlocks;
			}
			
			cutilDeviceSynchronize(); // Synchronize the device to ensure that all intermediate results are available
			
			// if sufficient work then do final (column-wise) reduction on GPU
			if (rows > REDUCE_GPU_THRESHOLD)
			{
				// reset to starting position and use it as an input
				d_input = deviceMemPointer;
				d_output = deviceMemPointer+rows; // re-use already allocated space as well for output.
				
#ifdef USE_PINNED_MEMORY
				copyHostToDevice(&tempResult[0], d_input, rows, this->m_environment->m_devices_CU[deviceID]->m_streams[0]);
#else
				copyHostToDevice(&tempResult[0], d_input, rows);
#endif
				
				std::tie(numThreads, numBlocks) = getNumBlocksAndThreads(rows, maxBlocks, maxThreads);
				
#ifdef USE_PINNED_MEMORY
				ExecuteReduceOnADevice(this->m_cuda_colwise_kernel, rows, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, deviceID, this->m_environment->m_devices_CU[deviceID]->m_streams[0]);
#else
				ExecuteReduceOnADevice(this->m_cuda_colwise_kernel, rows, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, deviceID);
#endif
				
#ifdef USE_PINNED_MEMORY
				copyDeviceToHost(&res, d_output, 1, this->m_environment->m_devices_CU[deviceID]->m_streams[0]);
#else
				copyDeviceToHost(&res, d_output, 1);
#endif
				
				cutilDeviceSynchronize();
			}
			else // do final reduction step on CPU instead
			{
				for (size_t r = 0; r < rows; r++)
					res = ReduceFuncColWise::CPU(res, tempResult[r]);
			}
			
			freeCudaMemory<T>(deviceMemPointer);
			
			return res;
		}
		
		
		/*!
		 *  Performs the 2D Reduction (First row-wise then column-wise) on an input
		 *  matrix by using \em CUDA backend. Returns a scalar result. The function
		 *  uses a variable number of devices, dividing the range of elemets equally
		 *  among the participating devices each reducing its part. The results are
		 *  then reduced themselves on the CPU.
		 */
		template <typename ReduceFuncRowWise, typename ReduceFuncColWise, typename CUDARowWise, typename CUDAColWise, typename CLKernel>
		typename ReduceFuncRowWise::Ret Reduce2D<ReduceFuncRowWise, ReduceFuncColWise, CUDARowWise, CUDAColWise, CLKernel>
		::reduceMultiple_CU(size_t numDevices, T &res, Matrix<T>& arg)
		{
			const size_t rows = arg.total_rows();
			const size_t cols = arg.total_cols();
			const size_t numRowsPerSlice = rows / numDevices;
			const size_t restRows = rows % numDevices;
			const size_t maxBlocks = this->m_selected_spec->GPUBlocks();
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			
			typename Matrix<T>::device_pointer_type_cu in_mem_p[MAX_GPU_DEVICES];
			
			T *deviceMemPointers[MAX_GPU_DEVICES];
			
			// Setup parameters
			size_t numThreads, numBlocks;
			std::tie(numThreads, numBlocks) = getNumBlocksAndThreads(cols, maxBlocks, maxThreads);
			
			// First create CUDA memory if not created already.
			for (size_t i = 0; i < numDevices; ++i)
			{
				cudaSetDevice(i);
				const size_t numRows = numRowsPerSlice + ((i == numDevices-1) ? restRows : 0);
				
				in_mem_p[i] = arg.updateDevice_CU((arg.getAddress() + i * numRowsPerSlice * cols), numRows * cols, i, AccessMode::None);
				
				size_t outSize = (i != 0) ? numRows * numBlocks : std::max(numRows * numBlocks, 2 * rows);
				
				// Manually allocate output memory in this case
				allocateCudaMemory<T>(&deviceMemPointers[i], outSize);
			}
			
			std::vector<T, malloc_allocator<T> > tempResult(rows);
			T *d_input = NULL, *d_output = NULL;
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				cudaSetDevice(i);
				
				Device_CU *device = this->m_environment->m_devices_CU[i];
				const size_t maxKernelsSupported = device->getNoConcurrentKernels();
				const size_t numRows = numRowsPerSlice + ((i == numDevices-1) ? restRows : 0);
				
				in_mem_p[i] = arg.updateDevice_CU((arg.getAddress() + i * numRowsPerSlice * cols), numRows * cols, i, AccessMode::Read);
				
				d_input = in_mem_p[i]->getDeviceDataPointer();
				d_output = deviceMemPointers[i];
				
				// First reduce all elements row-wise so that each row produces one element.
				for (size_t r = 0; r < numRows; r++)
				{
#ifdef USE_PINNED_MEMORY
					ExecuteReduceOnADevice(this->m_cuda_kernel, cols, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, i, device->m_streams[r % maxKernelsSupported]);
#else
					ExecuteReduceOnADevice(this->m_cuda_kernel, cols, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, i);
#endif

#ifdef USE_PINNED_MEMORY
					copyDeviceToHost(&tempResult[r+(numRowsPerSlice*i)], d_output, 1, this->m_environment->m_devices_CU[i]->m_streams[r % maxKernelsSupported]);
#else
					copyDeviceToHost(&tempResult[r+(numRowsPerSlice*i)], d_output, 1);
#endif
					if (r != numRows - 1)
					{
						d_input += cols;
						d_output += numBlocks;
					}
				}
			}
			
			// Synchronize all devices (?)
			this->m_environment->finishAll_CU(0, numDevices);
			
			// if sufficient work then do final (column-wise) reduction on GPU
			if (rows > REDUCE_GPU_THRESHOLD)
			{
				cudaSetDevice(this->m_environment->bestCUDADevID); // do it on a single GPU or a CPU, should not be that much work(?)
				
				// reset to starting position and use it as an input
				d_input = deviceMemPointers[0];
				d_output = deviceMemPointers[0]+rows; // re-use already allocated space as well for output.
				
#ifdef USE_PINNED_MEMORY
				copyHostToDevice(&tempResult[0], d_input, rows, this->m_environment->m_devices_CU[this->m_environment->bestCUDADevID]->m_streams[0]);
#else
				copyHostToDevice(&tempResult[0], d_input, rows);
#endif
				std::tie(numThreads, numBlocks) = getNumBlocksAndThreads(rows, maxBlocks, maxThreads);
				
#ifdef USE_PINNED_MEMORY
				ExecuteReduceOnADevice(this->m_cuda_colwise_kernel, rows, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, this->m_environment->bestCUDADevID, this->m_environment->m_devices_CU[m_environment->bestCUDADevID]->m_streams[0]);
#else
				ExecuteReduceOnADevice(this->m_cuda_colwise_kernel, rows, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, this->m_environment->bestCUDADevID);
#endif
				
#ifdef USE_PINNED_MEMORY
				copyDeviceToHost(&res, d_output, 1, this->m_environment->m_devices_CU[m_environment->bestCUDADevID]->m_streams[0]);
#else
				copyDeviceToHost(&res, d_output, 1);
#endif
				
				cutilDeviceSynchronize();
			}
			else // do final reduction step on CPU instead
			{
				for (size_t r = 1; r < rows; r++)
					res = ReduceFuncColWise::CPU(res, tempResult[r]);
			}
			
			// Free allocated memory on all devices, some pathetic issue, gives error when try to clear all poiters except first one
			for(size_t i = 0; i < numDevices; ++i)
			{
				cudaSetDevice(i);
				freeCudaMemory<T>(deviceMemPointers[i]);
			}
			
			return res;
		}
		
		
		/*!
		 *  Performs the 2D Reduction (First row-wise then column-wise) on an input
		 *  matrix by using \em CUDA backend. Returns a scalar result. The function
		 *  can be applied by any number of CUDA devices, thus internally calling the
		 *  \em reduceSingle_CL or \em reduceNumDevices_CL depending upon number of
		 *  CUDA devices specified/available.
		 */
		template<typename ReduceFuncRowWise, typename ReduceFuncColWise, typename CUDARowWise, typename CUDAColWise, typename CLKernel>
		typename ReduceFuncRowWise::Ret Reduce2D<ReduceFuncRowWise, ReduceFuncColWise, CUDARowWise, CUDAColWise, CLKernel>
		::CU(T &res, Matrix<T>& arg)
		{
			DEBUG_TEXT_LEVEL1("CUDA Reduce (2D): size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CU.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			if (numDevices <= 1)
				return this->reduceSingleThread_CU(this->m_environment->bestCUDADevID, res, arg);
				
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
				
			return this->reduceMultiple_CU(numDevices, res, arg);
		}
		
	}
}

#endif
