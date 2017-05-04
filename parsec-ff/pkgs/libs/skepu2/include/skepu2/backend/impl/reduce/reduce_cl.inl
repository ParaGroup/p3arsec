/*! \file reduce_cl.inl
*  \brief Contains the definitions of OpenCL specific member functions for the Reduce skeleton.
 */

#ifdef SKEPU_OPENCL

namespace skepu2
{
	namespace backend
	{
		/*!
		 *  A helper function that is used to call the actual kernel for reduction. Used by other functions to call the actual kernel
		 *  Internally, it just calls 2 kernels by setting their arguments. No synchronization is enforced.
		 *
		 *  \param size Size of the input array to be reduced.
		 *  \param numThreads Number of threads to be used for kernel execution.
		 *  \param numBlocks Number of blocks to be used for kernel execution.
		 *  \param inP OpenCL memory pointer to input array.
		 *  \param outP OpenCL memory pointer to output array.
		 *  \param kernel OpenCL kernel handle.
		 *  \param device OpenCL device handle.
		 */
		template<typename T, typename Kernel>
		void ExecuteReduceOnADevice(size_t deviceID, Kernel kernel, size_t size, size_t numThreads, size_t numBlocks, cl_mem inP, cl_mem outP)
		{
			DEBUG_TEXT_LEVEL1("OpenCL Reduce (two-phase kernel): size = " << size << ", numThreads = " << numThreads << ", numBlocks = " << numBlocks);
			
			const size_t globalWorkSize = numBlocks * numThreads;
			const size_t sharedMemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T);
			
			kernel(deviceID, numThreads, globalWorkSize, inP, outP, size, sharedMemSize);
			kernel(deviceID, numThreads, numThreads, outP, outP, numBlocks, sharedMemSize);
		}
		
		
		/*!
		 *  Performs the Reduction, either row-wise or column-wise, on a Matrix with \em OpenCL as backend.
		 *  Has an output parameter \em SkePU vector of reduction result. The function
		 *  uses only \em one device which is decided by a parameter. A Helper method.
		 */
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		Vector<typename ReduceFunc::Ret> &Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::reduceSingleThreadOneDim_CL(size_t deviceID, Vector<T> &res, Matrix<T> &arg)
		{
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			const size_t rows = arg.total_rows();
			const size_t cols = arg.total_cols();
			const size_t size = rows * cols;
			
			// Setup parameters
			size_t numBlocks, numThreads;
			std::tie(numThreads, numBlocks) = getNumBlocksAndThreads(cols, this->m_selected_spec->GPUBlocks(), this->m_selected_spec->GPUThreads());
			
			// Decide size of shared memory
			const size_t sharedMemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T);
			
			// Copies the elements to the device all at once, better(?)
			typename Matrix<T>::device_pointer_type_cl inMemP = arg.updateDevice_CL(arg.getAddress(), size, device, true);
			
			// Manually allocate output memory in this case, if only 1 block allocate for two
			cl_mem deviceMemPointer = cl_helpers::allocateDeviceMemory<T>(rows*numBlocks, device);
			cl_mem deviceInPointer = inMemP->getDeviceDataPointer();
			cl_mem d_input = deviceInPointer;
			cl_mem d_output = deviceMemPointer;
			
			// First reduce all elements row-wise so that each row produces one element.
			for (size_t r = 0; r < rows; r++)
			{
				if (r > 0)
				{
					cl_buffer_region inRegion  = cl_helpers::makeBufferRegion<T>(r * cols, cols);
					cl_buffer_region outRegion = cl_helpers::makeBufferRegion<T>(r * numBlocks, numBlocks);
					
					d_input  = clCreateSubBuffer(deviceInPointer,  CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &inRegion,  NULL);
					d_output = clCreateSubBuffer(deviceMemPointer, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &outRegion, NULL);
				}
				
				// execute the reduction for the given row
				ExecuteReduceOnADevice<T>(deviceID, CLKernel::reduce, cols, numThreads, numBlocks, d_input, d_output);
				
				// Now get the reduction result back for the given row
				cl_helpers::copyDeviceToHost<T>(&res[r], d_output, 1, device, 0);
				
				if (r > 0)
				{
					cl_helpers::freeDeviceMemory<T>(d_input);
					cl_helpers::freeDeviceMemory<T>(d_output);
				}
			}
			
			cl_helpers::freeDeviceMemory<T>(deviceMemPointer);
			
			return res;
		}
		
		
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		Vector<typename ReduceFunc::Ret> &Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::reduceMultipleOneDim_CL(size_t numDevices, Vector<T> &res, Matrix<T> &arg)
		{
			const size_t rows = arg.total_rows();
			const size_t cols = arg.total_cols();
			const size_t numRowsPerSlice = rows / numDevices;
			const size_t restRows = rows % numDevices;
			
			typename Matrix<T>::device_pointer_type_cl inMemP[MAX_GPU_DEVICES];
			cl_mem deviceMemPointers[MAX_GPU_DEVICES];
			
			// Setup parameters
			size_t numThreads, numBlocks;
			std::tie(numThreads, numBlocks) = getNumBlocksAndThreads(cols, this->m_selected_spec->GPUBlocks(), this->m_selected_spec->GPUThreads());
			
			// First create OpenCL memory if not created already.
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numRows = numRowsPerSlice + ((i == numDevices - 1) ? restRows : 0);
				const size_t baseIndex = i * numRowsPerSlice * cols;
				const size_t outSize = numRows * numBlocks;
				
				inMemP[i] = arg.updateDevice_CL((arg.getAddress() + baseIndex), numRows * cols, device, false);
				
				// Manually allocate output memory in this case
				deviceMemPointers[i] = cl_helpers::allocateDeviceMemory<T>(outSize, device);
			}
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numRows = numRowsPerSlice + ((i == numDevices - 1) ? restRows : 0);
				const size_t baseIndex = i * numRowsPerSlice * cols;
				
				inMemP[i] = arg.updateDevice_CL((arg.getAddress() + baseIndex), numRows * cols, device, true);
				
				cl_mem d_input = inMemP[i]->getDeviceDataPointer();
				cl_mem d_output = deviceMemPointers[i];
				cl_mem deviceInPointer = d_input;
				
				// First reduce all elements row-wise so that each row produces one element.
				for (size_t r = 0; r < numRows; r++)
				{
					if (r > 0)
					{
						cl_buffer_region inRegion  = cl_helpers::makeBufferRegion<T>(r * cols, cols);
						cl_buffer_region outRegion = cl_helpers::makeBufferRegion<T>(r * numBlocks, numBlocks);
						
						d_input  = clCreateSubBuffer(deviceInPointer,      CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &inRegion,  NULL);
						d_output = clCreateSubBuffer(deviceMemPointers[i], CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &outRegion, NULL);
					}
					
					// execute the reduction for the given row
					ExecuteReduceOnADevice<T>(i, CLKernel::reduce, cols, numThreads, numBlocks, d_input, d_output);
					
					// Now get the reduction result back for the given row
					cl_helpers::copyDeviceToHost<T>(&res[r+(numRowsPerSlice*i)], d_output, 1, device, 0);
					
					if (r > 0)
					{
						// Should delete the buffers allocated....
						cl_helpers::freeDeviceMemory<T>(d_input);
						cl_helpers::freeDeviceMemory<T>(d_output);
					}
				}
			}
			
			finishAll();
			
			for (size_t i = 0; i < numDevices; ++i)
				cl_helpers::freeDeviceMemory<T>(deviceMemPointers[i]);
			
			return res;
		}
		
		
		/*!
		 *  Performs the Reduction, either row-wise or column-wise, on a Matrix. Returns a \em SkePU vector of reduction result.
		 *  Using \em OpenCL as backend.
		 */
		template <typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		Vector<typename ReduceFunc::Ret> &Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::CL(Vector<T> &res, Matrix<T>& arg)
		{
			DEBUG_TEXT_LEVEL1("OpenCL Reduce (Matrix 1D): rows = " << arg.total_rows() << ", cols = " << arg.total_cols()
				<< ", maxDevices = " << this->m_selected_spec->devices() << ", maxBlocks = " << this->m_selected_spec->GPUBlocks()
				<< ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			if (numDevices <= 1)
				return this->reduceSingleThreadOneDim_CL(0, res, arg);
			
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			return this->reduceMultipleOneDim_CL(numDevices, res, arg);
		}
		
		
		/*!
		 *  Performs the Reduction on a range of elements with \em OpenCL as backend. Returns a scalar result. The function
		 *  uses only \em one device which is decided by a parameter. A Helper method.
		 */
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		template<typename Iterator>
		typename ReduceFunc::Ret Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::reduceSingle_CL(size_t deviceID, size_t size, T &res, Iterator arg)
		{
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			size_t numBlocks, numThreads;
			std::tie(numThreads, numBlocks) = getNumBlocksAndThreads(size, this->m_selected_spec->GPUBlocks(), this->m_selected_spec->GPUThreads());
			
			// Copy the elements to the device; create the output memory.
			DeviceMemPointer_CL<T> *inMemP = arg.getParent().updateDevice_CL(arg.getAddress(), size, device, true);
			DeviceMemPointer_CL<T> outMemP(&res, numBlocks, device);
			
			ExecuteReduceOnADevice<T>(deviceID, CLKernel::reduce, size, numThreads, numBlocks, inMemP->getDeviceDataPointer(), outMemP.getDeviceDataPointer());
			
			// Copy back result
			outMemP.changeDeviceData();
			outMemP.copyDeviceToHost(1);
			
			return res;
		}
		
		
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		template<typename Iterator>
		typename ReduceFunc::Ret Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::reduceMultiple_CL(size_t numDevices, size_t size, T &res, Iterator arg)
		{
			const size_t numElemPerSlice = size / numDevices;
			const size_t rest = size % numDevices;
			
			T result[MAX_GPU_DEVICES];
			std::vector<DeviceMemPointer_CL<T>*> outMemP(numDevices);
			
			// Setup parameters
			size_t numThreads[MAX_GPU_DEVICES];
			size_t numBlocks[MAX_GPU_DEVICES];
			
			// First create OpenCL memory if not created already.
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numElem = numElemPerSlice + ((i == numDevices - 1) ? rest : 0);
				std::tie(numThreads[i], numBlocks[i]) = getNumBlocksAndThreads(numElem, this->m_selected_spec->GPUBlocks(), this->m_selected_spec->GPUThreads());
				
				arg.getParent().updateDevice_CL(arg.getAddress() + i * numElemPerSlice, numElem, device, false);
				outMemP[i] = new DeviceMemPointer_CL<T>(&result[i], numBlocks[i], device);
			}
			
			// Create argument structs for all threads
			// Fill out argument struct with right information and start threads.
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numElem = numElemPerSlice + ((i == numDevices - 1) ? rest : 0);
				
				// Copies the elements to the device
				DeviceMemPointer_CL<T> *inMemP = arg.getParent().updateDevice_CL(arg.getAddress() + i * numElemPerSlice, numElem, device, true);
				
				ExecuteReduceOnADevice<T>(i, CLKernel::reduce, numElem, numThreads[i], numBlocks[i], inMemP->getDeviceDataPointer(), outMemP[i]->getDeviceDataPointer());
				
				// Copy back result
				outMemP[i]->changeDeviceData();
			}
			
			// Reduces results from each device on the CPU to yield the total result.
			for (size_t i = 0; i < numDevices; ++i)
			{
				outMemP[i]->copyDeviceToHost(1);
				res = ReduceFunc::CPU(res, result[i]);
				delete outMemP[i];
			}
			
			return res;
		}
		
		
		/*!
		 *  Performs the Reduction on a range of elements. Returns a scalar result. The function decides whether to perform
		 *  the reduction on one device, calling reduceSingle_CL(InputIterator inputBegin, InputIterator inputEnd, unsigned int deviceID) or
		 *  on multiple devices, calling reduceNumDevices_CL(InputIterator inputBegin, InputIterator inputEnd, size_t numDevices).
		 *  Using \em OpenCL as backend.
		 */
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		template<typename Iterator>
		typename ReduceFunc::Ret Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::CL(size_t size, T &res, Iterator arg)
		{
			DEBUG_TEXT_LEVEL1("OpenCL Reduce: size = " << size << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			if (numDevices <= 1)
				return this->reduceSingle_CL(0, size, res, arg);
			
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			return this->reduceMultiple_CL(numDevices, size, res, arg);
		}
		
		
		
		
		
		/*!
		 *  Performs the 2D Reduction (First row-wise then column-wise) on an input
		 *  matrix by using \em OpenCL backend. Returns a scalar result. The function
		 *  uses only \em one OpenCL device which is decided by a parameter.
		 */
		template<typename ReduceFuncRowWise, typename ReduceFuncColWise, typename CUDARowWise, typename CUDAColWise, typename CLKernel>
		typename ReduceFuncRowWise::Ret Reduce2D<ReduceFuncRowWise, ReduceFuncColWise, CUDARowWise, CUDAColWise, CLKernel>
		::reduceSingle_CL(size_t deviceID, T &res, Matrix<T>& arg)
		{
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			const size_t rows = arg.total_rows();
			const size_t cols = arg.total_cols();
			const size_t size = rows * cols;
			
			size_t numBlocks, numThreads;
			std::tie(numThreads, numBlocks) = getNumBlocksAndThreads(cols, this->m_selected_spec->GPUBlocks(), this->m_selected_spec->GPUThreads());
			
			// Decide size of shared memory
			const size_t sharedMemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T);
			
			// Copies the elements to the device all at once, better(?)
			typename Matrix<T>::device_pointer_type_cl in_mem_p = arg.updateDevice_CL(arg.getAddress(), size, device, true);
			
			std::vector<T> tempResult(rows);
			
			// Manually allocate output memory in this case, if only 1 block allocate for two
			cl_mem deviceMemPointer = cl_helpers::allocateDeviceMemory<T>(rows*((numBlocks>1)?numBlocks:2), device);
			cl_mem deviceInPointer = in_mem_p->getDeviceDataPointer();
			cl_mem d_input = deviceInPointer;
			cl_mem d_output = deviceMemPointer;
			
			// First reduce all elements row-wise so that each row produces one element.
			for (size_t r = 0; r < rows; r++)
			{
				if (r > 0)
				{
					cl_buffer_region inRegion  = cl_helpers::makeBufferRegion<T>(r * cols, cols);
					cl_buffer_region outRegion = cl_helpers::makeBufferRegion<T>(r * numBlocks, numBlocks);
					
					d_input  = clCreateSubBuffer(deviceInPointer, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &inRegion, NULL);
					d_output = clCreateSubBuffer(deviceMemPointer, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &outRegion, NULL);
				}
				
				// execute the reduction for the given row
				ExecuteReduceOnADevice<T>(deviceID, CLKernel::reduceRowWise, cols, numThreads, numBlocks, d_input, d_output);
				
				// Now get the reduction result back for the given row
				cl_helpers::copyDeviceToHost<T>(&tempResult[r], d_output, 1, device, 0);
			}
			
			// Do column-wise reduction, if sufficient work then do on GPU
			if (rows > REDUCE_GPU_THRESHOLD)
			{
				clFinish(device->getQueue());
				
				cl_buffer_region region = cl_helpers::makeBufferRegion<T>(rows, rows);
				d_output = clCreateSubBuffer(deviceMemPointer, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, NULL);
				d_input = deviceMemPointer;
				
				cl_helpers::copyHostToDevice<T>(&tempResult[0], d_input, rows, device, 0);
				
				// execute the reduction for the resulting row
				std::tie(numThreads, numBlocks) = getNumBlocksAndThreads(rows, this->m_selected_spec->GPUBlocks(), this->m_selected_spec->GPUThreads());
				ExecuteReduceOnADevice<T>(deviceID, CLKernel::reduceColWise, rows, numThreads, numBlocks, d_input, d_output);
				
				clFinish(device->getQueue());
				
				cl_helpers::copyDeviceToHost<T>(&res, d_output, 1, device, 0);
			}
			else // do final reduction step on CPU instead
			{
				for (size_t r = 0; r < rows; r++)
					res = ReduceFuncColWise::CPU(res, tempResult[r]);
			}
			
			cl_helpers::freeDeviceMemory<T>(deviceMemPointer);
			
			return res;
		}
		
		
		/*!
		 *  Performs the 2D Reduction (First row-wise then column-wise) on an input
		 *  matrix by using \em OpenCL backend. Returns a scalar result. The function
		 *  uses a variable number of devices, dividing the range of elemets equally
		 *  among the participating devices each reducing its part. The results are
		 *  then reduced themselves on the CPU.
		 */
		template <typename ReduceFuncRowWise, typename ReduceFuncColWise, typename CUDARowWise, typename CUDAColWise, typename CLKernel>
		typename ReduceFuncRowWise::Ret Reduce2D<ReduceFuncRowWise, ReduceFuncColWise, CUDARowWise, CUDAColWise, CLKernel>
		::reduceNumDevices_CL(size_t numDevices, T &res, Matrix<T>& arg)
		{
			const size_t rows = arg.total_rows();
			const size_t cols = arg.total_cols();
			const size_t numRowsPerSlice = rows / numDevices;
			const size_t restRows = rows % numDevices;
			
			typename Matrix<T>::device_pointer_type_cl in_mem_p[MAX_GPU_DEVICES];
			cl_mem deviceMemPointers[MAX_GPU_DEVICES];
			
			// Setup parameters
			size_t numThreads, numBlocks;
			std::tie(numThreads, numBlocks) = getNumBlocksAndThreads(cols, this->m_selected_spec->GPUBlocks(), this->m_selected_spec->GPUThreads());
			
			// First create OpenCL memory if not created already.
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numRows = numRowsPerSlice + ((i == numDevices-1) ? restRows : 0);
				
				in_mem_p[i] = arg.updateDevice_CL(arg.getAddress() + i * numRowsPerSlice * cols, numRows * cols, device, false);
				
				// For first device as later we may re-use this storage to do final reduction on GPU 0, make it at least that much large
				size_t outSize = (i != 0) ? numRows * numBlocks : std::max(numRows * numBlocks, 2 * rows);
				
				// Manually allocate output memory in this case
				deviceMemPointers[i] = cl_helpers::allocateDeviceMemory<T>(outSize, device);
			}
			
			std::vector<T> tempResult(rows);
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numRows = numRowsPerSlice + ((i == numDevices-1) ? restRows : 0);
				
				in_mem_p[i] = arg.updateDevice_CL((arg.getAddress()+i*numRowsPerSlice*cols), numRows*cols, device, true);
				
				cl_mem d_input = in_mem_p[i]->getDeviceDataPointer();
				cl_mem d_output = deviceMemPointers[i];
				cl_mem deviceInPointer = d_input;
				
				// First reduce all elements row-wise so that each row produces one element.
				for (size_t r = 0; r < numRows; r++)
				{
					if (r > 0)
					{
						cl_buffer_region inRegion  = cl_helpers::makeBufferRegion<T>(r * cols, cols);
						cl_buffer_region outRegion = cl_helpers::makeBufferRegion<T>(r * numBlocks, numBlocks);
						
						d_input  = clCreateSubBuffer(deviceInPointer,      CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &inRegion,  NULL);
						d_output = clCreateSubBuffer(deviceMemPointers[i], CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &outRegion, NULL);
					}
					
					// execute the reduction for the given row
					ExecuteReduceOnADevice<T>(i, CLKernel::reduceRowWise, cols, numThreads, numBlocks, d_input, d_output);
					
					// Now get the reduction result back for the given row
					cl_helpers::copyDeviceToHost<T>(&tempResult[r + (numRowsPerSlice * i)], d_output, 1, device, 0);
				}
			}
			
			this->m_environment->finishAll();
			
			// if sufficient work then do final (column-wise) reduction on GPU 0
			if (rows > REDUCE_GPU_THRESHOLD)
			{
				Device_CL *device = this->m_environment->m_devices_CL[0];
				
				cl_buffer_region region = cl_helpers::makeBufferRegion<T>(rows, rows);
				cl_mem d_output = clCreateSubBuffer(deviceMemPointers[0], CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, NULL);
				cl_mem d_input = deviceMemPointers[0];
				
				cl_helpers::copyHostToDevice<T>(&tempResult[0], d_input, rows, device, 0);
				
				// execute the reduction for the resulting row
				std::tie(numThreads, numBlocks) = getNumBlocksAndThreads(rows, this->m_selected_spec->GPUBlocks(), this->m_selected_spec->GPUThreads());
				ExecuteReduceOnADevice<T>(0, CLKernel::reduceColWise, rows, numThreads, numBlocks, d_input, d_output);
				
				clFinish(device->getQueue());
				cl_helpers::copyDeviceToHost<T>(&res, d_output, 1, device, 0);
			}
			else // do final reduction step on CPU instead
			{
				for (size_t r = 0; r < rows; r++)
					res = ReduceFuncColWise::CPU(res, tempResult[r]);
			}
			
			// Free allocated memory on all devices
			for (size_t i = 0; i < numDevices; ++i)
				cl_helpers::freeDeviceMemory<T>(deviceMemPointers[i]);
			
			return res;
		}
		
		
		/*!
		 *  Performs the 2D Reduction (First row-wise then column-wise) on an input
		 *  matrix by using \em OpenCL backend. Returns a scalar result. The function
		 *  can be applied by any number of OpenCL devices, thus internally calling the
		 *  \em reduceSingle_CL or \em reduceNumDevices_CL depending upon number of
		 *  OpenCL devices specified/available.
		 */
		template<typename ReduceFuncRowWise, typename ReduceFuncColWise, typename CUDARowWise, typename CUDAColWise, typename CLKernel>
		typename ReduceFuncRowWise::Ret Reduce2D<ReduceFuncRowWise, ReduceFuncColWise, CUDARowWise, CUDAColWise, CLKernel>
		::CL(T &res, Matrix<T>& arg)
		{
			DEBUG_TEXT_LEVEL1("OpenCL Reduce (2D): size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			if (numDevices <= 1)
				return this->reduceSingle_CL(0, res, arg);
			
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			return this->reduceNumDevices_CL(numDevices, res, arg);
		}

	}
}

#endif
