/*! \file mapoverlap_cu.inl
 *  \brief Contains the definitions of CUDA specific member functions for the MapOverlap skeleton.
 */

#ifdef SKEPU_CUDA

#include <cuda.h>

namespace skepu2
{
	namespace backend
	{
		/*!
		 *  Applies the MapOverlap skeleton to a range of elements specified by iterators. Result is saved to a seperate output range.
		 *  The function uses only \em one device which is decided by a parameter. Using \em CUDA as backend.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<template<class> class Container, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::mapOverlapSingleThread_CU(size_t deviceID, Container<Ret>& res, Container<T>& arg, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			cudaSetDevice(deviceID);
			
			// Setup parameters
			const size_t n = arg.size();
			const size_t overlap = this->m_overlap;
			
			// Constructs a wrap vector, which is used if cyclic edge policy is specified.
			auto inputEndMinusOverlap = arg.end() - overlap;
			std::vector<T> wrap(2 * overlap);
			
			DeviceMemPointer_CU<T> *wrapMemP = NULL;
			
			if (this->m_edge == Edge::Cyclic)
			{
				// Just update here to get latest values back.
				arg.updateHost();
				
				for (size_t i = 0; i < overlap; ++i)
				{
					wrap[i] = inputEndMinusOverlap(i);
					wrap[overlap+i] = arg(i);
				}
				// Copy wrap vector to device only if is is cyclic policy as otherwise it is not used.
				wrapMemP = new DeviceMemPointer_CU<T>(&wrap[0], wrap.size(), m_environment->m_devices_CU.at(deviceID));
				wrapMemP->copyHostToDevice();
			}
			else // construct a very naive dummy
				wrapMemP = new DeviceMemPointer_CU<T>(NULL, 1, m_environment->m_devices_CU.at(deviceID));
			
			size_t numThreads = std::min(this->m_selected_spec->GPUThreads(), n);
			
			//----- START: error check for possible high overlap value which can bloat the shared memory ------//
			
			if (numThreads<overlap)
				SKEPU_ERROR("Overlap is higher than maximum threads available. MapOverlap Aborted!!!\n");
			
			size_t maxShMem = this->m_environment->m_devices_CU.at(deviceID)->getSharedMemPerBlock() / sizeof(T) - SHMEM_SAFITY_BUFFER; // little buffer for other usage
			size_t orgThreads = numThreads;
			numThreads = (numThreads + 2 * overlap < maxShMem) ? numThreads : maxShMem - 2 * overlap;
			
			if (orgThreads != numThreads && (numThreads < 8 || numThreads < overlap)) // if changed then atleast 8 threads should be there
				SKEPU_ERROR("Error: Possibly overlap is too high for operation to be successful on this GPU. MapOverlap Aborted!!!\n");
			
			//----- END: error check for possible high overlap value which can bloat the shared memory ------//
			
			size_t numBlocks = std::max((size_t)1, std::min( (n/numThreads + (n%numThreads == 0 ? 0:1)), this->m_selected_spec->GPUBlocks()));
			size_t sharedMemSize = sizeof(T) * (numThreads+2*overlap);
			
			// Copy elements to device and allocate output memory.
			auto inMemP = arg.updateDevice_CU(arg.getAddress(), n, deviceID, AccessMode::Read);
			auto outMemP = res.updateDevice_CU(res.getAddress(), n, deviceID, AccessMode::Write);
			auto anyMemP = std::make_tuple(get<AI, CallArgs...>(args...).cudaProxy(deviceID, MapOverlapFunc::anyAccessMode[AI])...);
			
#ifdef USE_PINNED_MEMORY
			this->m_cuda_kernel<<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>
#else
			this->m_cuda_kernel<<<numBlocks, numThreads, sharedMemSize>>>
#endif
			(
				inMemP->getDeviceDataPointer(),
				std::get<AI>(anyMemP).second...,
				get<CI, CallArgs...>(args...)...,
				outMemP->getDeviceDataPointer(),
				wrapMemP->getDeviceDataPointer(),
				n, 0, n,
				static_cast<int>(this->m_edge),
				this->m_pad, this->m_overlap
			);
			
			// Make sure the data is marked as changed by the device
			outMemP->changeDeviceData();
			
#ifdef TUNER_MODE
			cudaDeviceSynchronize();
#endif
		
			delete wrapMemP;
		}
		
		
		/*!
		 *  Applies the MapOverlap skeleton to a range of elements specified by iterators. Result is saved to a seperate output range.
		 *  The function uses a variable number of devices, dividing the range of elemets equally among the participating devices each mapping
		 *  its part. Using \em OpenCL as backend.
		 */
		template <typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<template<class> class Container, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::mapOverlapNumDevices_CU(size_t numDevices, Container<Ret>& res, Container<T>& arg, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("MAPOVERLAP CUDA\n")
			
			const size_t totalNumElements = arg.size();
			const size_t overlap = this->m_overlap;
			size_t numElemPerSlice = totalNumElements / numDevices;
			size_t rest = totalNumElements % numDevices;
			
			//Need to get new values from other devices so that the overlap between devices is up to date.
			//Bad for performance since whole vector needs to be copied, fix so that only overlap is fetched and updated.
		//	arg.updateHostAndInvalidateDevice();
			
			auto inputEndMinusOverlap = arg.end() - overlap;
			std::vector<T> wrap(2 * overlap);
			if (this->m_edge == Edge::Cyclic)
			{
				// Just update here to get latest values back.
				arg.updateHost();
				
				for (size_t i = 0; i < overlap; ++i)
				{
					wrap[i] = inputEndMinusOverlap(i);
					wrap[overlap+i] = arg(i);
				}
			}
			
			DeviceMemPointer_CU<T> *in_mem_p[MAX_GPU_DEVICES];
			DeviceMemPointer_CU<Ret> *out_mem_p[MAX_GPU_DEVICES];
			DeviceMemPointer_CU<T> *wrap_mem_p[MAX_GPU_DEVICES];
			
			size_t numBlocks[MAX_GPU_DEVICES];
			size_t numThreads[MAX_GPU_DEVICES];
			size_t n[MAX_GPU_DEVICES];
			
			// First create CUDA memory if not created already.
			for (size_t i = 0; i < numDevices; ++i)
			{
				const size_t numElem = numElemPerSlice + ((i == numDevices - 1) ? rest : 0);
				n[i] = numElem;
				
				if (i == 0)
				{
					in_mem_p[i] = arg.updateDevice_CU(arg.getAddress(), numElem+overlap, i, AccessMode::None);
					n[i] = numElem + overlap;
				}
				else if (i == numDevices-1)
				{
					in_mem_p[i] = arg.updateDevice_CU(arg.getAddress() + i * numElemPerSlice - overlap, numElem + overlap, i, AccessMode::None);
					n[i] = numElem + overlap;
				}
				else
				{
					in_mem_p[i] = arg.updateDevice_CU(arg.getAddress() + i * numElemPerSlice - overlap, numElem + 2 * overlap, i, AccessMode::None);
					n[i] = numElem + 2 * overlap;
				}
				
				out_mem_p[i] = res.updateDevice_CU(res.getAddress() + i * numElemPerSlice, numElem, i, AccessMode::None);
				
				// wrap vector, don't copy just allocates space onf CUDA device.
				wrap_mem_p[i] = new DeviceMemPointer_CU<T>(&wrap[0], wrap.size(), m_environment->m_devices_CU.at(i));
				
				numThreads[i] = std::min(this->m_selected_spec->GPUThreads(), n[i]);
				
				//----- START: error check for possible high overlap value which can bloat the shared memory ------//
				if (numThreads[i]<overlap)
					SKEPU_ERROR("Overlap is higher than maximum threads available. MapOverlap Aborted");
				
				size_t maxShMem = this->m_environment->m_devices_CU.at(i)->getSharedMemPerBlock() / sizeof(T) - SHMEM_SAFITY_BUFFER; // little buffer for other usage
				size_t orgThreads = numThreads[i];
				numThreads[i] = (numThreads[i] + 2 * overlap < maxShMem) ? numThreads[i] : maxShMem - 2 * overlap;
				
				if (orgThreads != numThreads[i] && (numThreads[i]<8 || numThreads[i]<overlap)) // if changed then atleast 8 threads should be there
					SKEPU_ERROR("Possibly overlap is too high for operation to be successful on this GPU. MapOverlap Aborted");
				
				//----- END: error check for possible high overlap value which can bloat the shared memory ------//
				
				numBlocks[i] = std::max<size_t>(1, std::min<size_t>(n[i] / numThreads[i] + (n[i] % numThreads[i] == 0) ? 0 : 1, this->m_selected_spec->GPUBlocks()));
			}
			
			// parameters
			size_t out_offset, out_numelements;
			
			// Fill out argument struct with right information and start threads.
			for (size_t i = 0; i < numDevices; ++i)
			{
				cudaSetDevice(i);
				
				size_t numElem;
				if (i == numDevices-1)
					numElem = numElemPerSlice+rest;
				else
					numElem = numElemPerSlice;
				
				if (this->m_edge == Edge::Cyclic)
				{
					// Copy actual wrap vector to device only if it is cyclic policy. Otherwise it is just an extra overhead.
					wrap_mem_p[i]->copyHostToDevice(); // it take the main execution time
				}
				
				// Copy elemets to device and set other kernel parameters depending on which device it is, first, last or a middle device.
				if (i == 0)
				{
					in_mem_p[i] = arg.updateDevice_CU(arg.getAddress() + i * numElemPerSlice, numElem + overlap, i, AccessMode::Read);
					out_offset = 0;
					out_numelements = numElem;
				}
				else if (i == numDevices-1)
				{
					in_mem_p[i] = arg.updateDevice_CU(arg.getAddress() + i * numElemPerSlice - overlap, numElem+overlap, i, AccessMode::Read);
					out_offset = overlap;
					out_numelements = numElem;
				}
				else
				{
					in_mem_p[i] = arg.updateDevice_CU(arg.getAddress() + i * numElemPerSlice - overlap, numElem + 2 * overlap, i, AccessMode::Read);
					out_offset = overlap;
					out_numelements = numElem;
				}
				
				out_mem_p[i] = res.updateDevice_CU(res.getAddress() + i * numElemPerSlice, numElem, i, AccessMode::Write, true);
				
				auto anyMemP = std::make_tuple(get<AI, CallArgs...>(args...).cudaProxy(i, MapOverlapFunc::anyAccessMode[AI])...);
				
				const size_t sharedMemSize = sizeof(T) * (numThreads[i] + 2 * overlap);
				
#ifdef USE_PINNED_MEMORY
					this->m_cuda_kernel<<<numBlocks[i], numThreads[i], sharedMemSize, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>
#else
					this->m_cuda_kernel<<<numBlocks[i], numThreads[i], sharedMemSize>>>
#endif
					(
						in_mem_p[i]->getDeviceDataPointer(),
						std::get<AI>(anyMemP).second...,
						get<CI, CallArgs...>(args...)...,
						out_mem_p[i]->getDeviceDataPointer(),
						wrap_mem_p[i]->getDeviceDataPointer(),
						n[i], out_offset, out_numelements,
						static_cast<int>(this->m_edge),
						this->m_pad, this->m_overlap
					);
				
				// Make sure the data is marked as changed by the device
				out_mem_p[i]->changeDeviceData();
			}
			
			// to properly de-allocate the memory
			for (size_t i = 0; i < numDevices; ++i)
				delete wrap_mem_p[i];
			
			cudaSetDevice(m_environment->bestCUDADevID);
		}
		
		
		/*!
		 *  Performs the MapOverlap on a range of elements. With a seperate output range. The function decides whether to perform the MapOverlap on one device.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<template<class> class Container, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::vector_CUDA(Container<Ret>& res, Container<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("CUDA MapOverlap 1D Matrix: size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CU.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			if (numDevices <= 1)
				return this->mapOverlapSingleThread_CU(0, res, arg, ai, ci, args...);
			else
			
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
				
				return this->mapOverlapNumDevices_CU(numDevices, res, arg, ai, ci, args...);
		}
		
		
		/*!
		 * For Matrix overlap, we need to check whether overlap configuration is runnable considering total size of shared memory available on that system.
		 * This method is a helper funtion doing that. It is called by another helper \p getThreadNumber_CU() method.
		 *
		 * \param numThreads Number of threads in a thread block.
		 * \param deviceID The device ID.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<typename T>
		bool MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::sharedMemAvailable_CU(size_t &numThreads, size_t deviceID)
		{
			const size_t overlap = this->m_overlap;
			const size_t maxShMem = this->m_environment->m_devices_CU.at(deviceID)->getSharedMemPerBlock() / sizeof(T) - SHMEM_SAFITY_BUFFER;
			size_t orgThreads = numThreads;
			numThreads = (numThreads + 2 * overlap < maxShMem) ? numThreads : maxShMem - 2 * overlap;
			
			if (orgThreads == numThreads) // return true when nothing changed because of overlap constraint
				return true;
			
			if (numThreads < 8) // not original numThreads then atleast 8 threads should be there
				SKEPU_ERROR("CUDA MapOverlap: Possibly overlap is too high for operation to be successful on this GPU.");
			
			return false;
		}
		
		
		/*!
		 * Helper method used for calculating optimal thread count. For row- or column-wise overlap,
		 * it determines a thread block size with perfect division of problem size.
		 *
		 * \param width The problem size.
		 * \param numThreads Number of threads in a thread block.
		 * \param deviceID The device ID.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<typename T>
		size_t MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::getThreadNumber_CU(size_t width, size_t &numThreads, size_t deviceID)
		{
			// first check whether shared memory would be ok for this numThreads. Changes numThreads accordingly
			if (!sharedMemAvailable_CU<T>(numThreads, deviceID) && numThreads < 1)
				SKEPU_ERROR("CUDA MapOverlap: Too low overlap size to continue.");
			
			if (width % numThreads == 0)
				return width / numThreads;
			
			// decreament numThreads and see which one is a perfect divisor
			for (size_t i = numThreads - 1; i >= 1; i--)
			{
				if (width % numThreads == 0)
					return width / numThreads;
			}
			
			return 0;
		}
		
		
		/*!
		 *  Performs the column-wise MapOverlap on a range of elements, using 1 GPU, on the \em CUDA with a seperate output range.
		 *  Used internally by other methods to apply column-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::mapOverlapSingleThread_CU_Col(size_t deviceID, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			cudaSetDevice(deviceID);
			
			const size_t n = arg.size();
			const size_t overlap = this->m_overlap;
			const size_t colWidth = arg.total_rows();
			const size_t numCols = arg.total_cols();
			size_t numThreads = this->m_selected_spec->GPUThreads();
			size_t trdsize = colWidth;
			size_t blocksPerCol = 1;
			
			if (colWidth > numThreads)
			{
				size_t tmp = getThreadNumber_CU<T>(colWidth, numThreads, deviceID);
				
				if (tmp < 1 || tmp * numCols > this->m_selected_spec->GPUBlocks())
					SKEPU_ERROR("CUDA MapOverlap: Operation is larger than maximum block size! colWidth: " << colWidth << ", numThreads: " << numThreads);
				blocksPerCol = tmp;
				trdsize = colWidth / blocksPerCol;
				
				if (trdsize < overlap)
					SKEPU_ERROR("CUDA MapOverlap: Cannot execute overlap with current overlap width.");
			}
			
			std::vector<T> wrap(2 * overlap * numCols);
			DeviceMemPointer_CU<T> wrap_mem_p(&wrap[0], wrap.size(), m_environment->m_devices_CU.at(deviceID));
			
			if (this->m_edge == Edge::Cyclic)
			{
				arg.updateHost();
				auto inputBeginTemp = arg.begin();
				
				for (size_t col = 0; col < numCols; col++, inputBeginTemp++)
				{
					auto inputEndMinusOverlap = inputBeginTemp + numCols * (colWidth - 1) - (overlap - 1) * numCols;
					for (size_t i = 0; i < overlap; ++i)
					{
						wrap[i + col * 2 * overlap] = inputEndMinusOverlap(i * numCols);
						wrap[overlap + i + col * 2 * overlap] = inputBeginTemp(i * numCols);
					}
				}
				wrap_mem_p.copyHostToDevice();
			}
			
			numThreads = trdsize; //std::min(maxThreads, rowWidth);
			size_t numBlocks = std::max<size_t>(1, std::min(blocksPerCol * numCols, this->m_selected_spec->GPUBlocks()));
			size_t sharedMemSize = sizeof(T) * (numThreads + 2 * overlap);
			
			DEBUG_TEXT_LEVEL1("CUDA MapOverlap Single: numBlocks = " << numBlocks << ", numThreads = " << numThreads);
			
			// Copy elements to device and allocate output memory.
			auto in_mem_p = arg.updateDevice_CU(arg.getAddress(), n, deviceID, AccessMode::Read, false);
			auto out_mem_p = res.updateDevice_CU(res.getAddress(), n, deviceID, AccessMode::Write, false);
			auto anyMemP = std::make_tuple(get<AI, CallArgs...>(args...).cudaProxy(deviceID, MapOverlapFunc::anyAccessMode[AI])...);
			
#ifdef USE_PINNED_MEMORY
			this->m_cuda_colwise_kernel<<<numBlocks, numThreads, sharedMemSize, this->m_environment->m_devices_CU.at(deviceID)->m_streams[0]>>>
#else
			this->m_cuda_colwise_kernel<<<numBlocks, numThreads, sharedMemSize>>>
#endif
			(
				in_mem_p->getDeviceDataPointer(),
				std::get<AI>(anyMemP).second...,
				get<CI, CallArgs...>(args...)...,
				out_mem_p->getDeviceDataPointer(),
				wrap_mem_p.getDeviceDataPointer(),
				n, 0, n,
				static_cast<int>(this->m_edge),
				this->m_pad, this->m_overlap,
				blocksPerCol, numCols, colWidth
			);
			
			// Make sure the data is marked as changed by the device
			out_mem_p->changeDeviceData();
		}
		
		
		/*!
		 *  Performs the column-wise MapOverlap on a range of elements, using multiple GPUs, on the \em CUDA with a seperate output range.
		 *  Used internally by other methods to apply column-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::mapOverlapMultiThread_CU_Col(size_t numDevices, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			const size_t totalElems = arg.size();
			const size_t maxBlocks = this->m_selected_spec->GPUBlocks();
			size_t maxThreads = this->m_selected_spec->GPUThreads();
			const size_t overlap = this->m_overlap;
			const size_t colWidth = arg.total_rows();
			const size_t numCols = arg.total_cols();
			const size_t numRowsPerSlice = colWidth / numDevices;
			const size_t numElemPerSlice = numRowsPerSlice * numCols;
			const size_t restRows = colWidth % numDevices;
			
			auto inputEndTemp = arg.end();
			auto inputBeginTemp = arg.begin();
			auto inputEndMinusOverlap = arg.end() - overlap;
			
			std::vector<T> wrapStartDev(overlap*numCols);
			std::vector<T> wrapEndDev(overlap*numCols);
			
			if (this->m_edge == Edge::Cyclic)
			{
				size_t stride = numCols;
			
				// Just update here to get latest values back.
				arg.updateHost();
				for (size_t col=0; col< numCols; col++)
				{
					inputEndTemp = inputBeginTemp+(numCols*(colWidth-1));
					inputEndMinusOverlap = (inputEndTemp - (overlap-1)*stride);
					
					for (size_t i = 0; i < overlap; ++i)
					{
						wrapStartDev[i+(col*overlap)] = inputEndMinusOverlap(i*stride);
						wrapEndDev[i+(col*overlap)] = inputBeginTemp(i*stride);
					}
					inputBeginTemp++;
				}
			}
			
			DeviceMemPointer_CU<T> *in_mem_p[MAX_GPU_DEVICES];
			DeviceMemPointer_CU<Ret> *out_mem_p[MAX_GPU_DEVICES];
			DeviceMemPointer_CU<T> *wrapStartDev_mem_p[MAX_GPU_DEVICES];
			DeviceMemPointer_CU<T> *wrapEndDev_mem_p[MAX_GPU_DEVICES];
			
			size_t trdsize[MAX_GPU_DEVICES];
			size_t blocksPerCol[MAX_GPU_DEVICES];
			
			// First create CUDA memory if not created already.
			for (size_t i = 0; i < numDevices; ++i)
			{
				const size_t numRows = numRowsPerSlice + ((i == numDevices - 1) ? restRows : 0);
				
				trdsize[i] = numRows;
				blocksPerCol[i] = 1;
				
				if (numRows > maxThreads)
				{
					size_t tmp = getThreadNumber_CU<T>(numRows, maxThreads, i);
					if (tmp < 1 || tmp * numCols > maxBlocks)
						SKEPU_ERROR("CUDA MapOverlap: Col width is larger than maximum thread size: " << colWidth << " " << maxThreads);
					blocksPerCol[i] = tmp;
					trdsize[i] = numRows / blocksPerCol[i];
					
					if (trdsize[i] < overlap)
						SKEPU_ERROR("CUDA MapOverlap: Cannot execute overlap with current overlap width");
				}
				
				size_t overlapElems = overlap * numCols;
				
				if (i == 0)                 in_mem_p[i] = arg.updateDevice_CU(arg.getAddress() + i * numElemPerSlice,                (numRows + overlap) * numCols,     i, AccessMode::None, false);
				else if (i == numDevices-1) in_mem_p[i] = arg.updateDevice_CU(arg.getAddress() + i * numElemPerSlice - overlapElems, (numRows + overlap) * numCols,     i, AccessMode::None, false);
				else                        in_mem_p[i] = arg.updateDevice_CU(arg.getAddress() + i * numElemPerSlice - overlapElems, (numRows + 2 * overlap) * numCols, i, AccessMode::None, false);
				
				out_mem_p[i] = res.updateDevice_CU(res.getAddress() + i * numElemPerSlice, numRows * numCols, i, AccessMode::None, false);
				
				// wrap vector, don't copy just allocates space on CUDA device.
				wrapStartDev_mem_p[i] = new DeviceMemPointer_CU<T>(&wrapStartDev[0], wrapStartDev.size(), this->m_environment->m_devices_CU.at(i));
				wrapEndDev_mem_p[i]   = new DeviceMemPointer_CU<T>(&wrapEndDev[0],   wrapEndDev.size(),   this->m_environment->m_devices_CU.at(i));
			}
			
			// we will divide the computation row-wise... copy data and do operation
			for (size_t i = 0; i < numDevices; i++)
			{
				const size_t numRows = numRowsPerSlice + ((i == numDevices - 1) ? restRows : 0);
				const size_t numElems = numRows * numCols;
				size_t n, in_offset, overlapElems = overlap * numCols;
				
				// Copy elemets to device and set other kernel parameters depending on which device it is, first, last or a middle device.
				if (i == 0)
				{
					if (this->m_edge == Edge::Cyclic)
						wrapStartDev_mem_p[i]->copyHostToDevice(); // copy wrap as it will be used partially, may optimize further by placing upper and lower overlap in separate buffers for muultiple devices?
					
					in_mem_p[i] = arg.updateDevice_CU(arg.getAddress() + i * numElemPerSlice, (numRows + overlap) * numCols, i, AccessMode::Read, false);
					in_offset = 0;
					n = numElems; //numElem+overlap;
				}
				else if (i == numDevices-1)
				{
					if (this->m_edge == Edge::Cyclic)
						wrapEndDev_mem_p[i]->copyHostToDevice(); // copy wrap as it will be used partially, may optimize further by placing upper and lower overlap in separate buffers for muultiple devices?
					
					in_mem_p[i] = arg.updateDevice_CU(arg.getAddress() + i * numElemPerSlice - overlapElems, (numRows + overlap) * numCols, i, AccessMode::Read, false);
					in_offset = overlapElems;
					n = numElems; //+overlap;
				}
				else
				{
					in_mem_p[i] = arg.updateDevice_CU(arg.getAddress() + i * numElemPerSlice-overlapElems, (numRows + 2 * overlap) * numCols, i, AccessMode::Read, false);
					in_offset = overlapElems;
					n = numElems; //+2*overlap;
				}
				
				out_mem_p[i] = res.updateDevice_CU(res.getAddress() + i * numElemPerSlice, numRows * numCols, i, AccessMode::Write, false, true);
				
				auto anyMemP = std::make_tuple(get<AI, CallArgs...>(args...).getParent().updateDevice_CU(get<AI, CallArgs...>(args...).getAddress(),
					get<AI, CallArgs...>(args...).getParent().size(), i, MapOverlapFunc::anyAccessMode[AI])...);
				
				const size_t numThreads = trdsize[i]; //std::min(maxThreads, rowWidth);
				const size_t numBlocks = std::max<size_t>(1, std::min(blocksPerCol[i] * numCols, maxBlocks));
				const size_t sharedMemSize = sizeof(T) * (numThreads + 2 * overlap);
				
				cudaSetDevice(i);
				
				int deviceType;
				if (i == 0) deviceType = -1;
				else if (i == numDevices-1) deviceType = 1;
				else deviceType = 0;
				
#ifdef USE_PINNED_MEMORY
				this->m_cuda_colwise_multi_kernel<<<numBlocks, numThreads, sharedMemSize, this->m_environment->m_devices_CU.at(i)->m_streams[0]>>>
#else
				this->m_cuda_colwise_multi_kernel<<<numBlocks, numThreads, sharedMemSize>>>
#endif
				(
					in_mem_p[i]->getDeviceDataPointer(),
					std::get<AI>(anyMemP).second...,
					get<CI, CallArgs...>(args...)...,
					out_mem_p[i]->getDeviceDataPointer(),
					wrapStartDev_mem_p[i]->getDeviceDataPointer(),
					n, in_offset, n,
					static_cast<int>(this->m_edge),
					deviceType,
					this->m_pad, this->m_overlap,
					blocksPerCol[i],
					numCols, numRows
				);
				
				// Make sure the data is marked as changed by the device
				out_mem_p[i]->changeDeviceData();
			}
			
			// to properly de-allocate the memory
			for (size_t i = 0; i < numDevices; ++i)
			{
				delete wrapStartDev_mem_p[i];
				delete wrapEndDev_mem_p[i];
			}
			
			cudaSetDevice(m_environment->bestCUDADevID);
		}
		
		
		/*!
		 *  Performs the MapOverlap on a range of elements. With a seperate output range. The function decides whether to perform the MapOverlap on one device.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::colwise_CUDA(Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("CUDA MapOverlap 1D Matrix: size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CU.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			if (numDevices <= 1)
				return this->mapOverlapSingleThread_CU_Col(0, res, arg, ai, ci, args...);
			else
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
				return this->mapOverlapMultiThread_CU_Col(numDevices, res, arg, ai, ci, args...);
		}
		
		
		/*!
		*  Performs the row-wise MapOverlap on a range of elements, using 1 GPU, on the \em CUDA with a seperate output range.
		*  Used internally by other methods to apply row-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::mapOverlapSingleThread_CU_Row(size_t deviceID, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			const size_t n = arg.size();
			const size_t maxBlocks = this->m_selected_spec->GPUBlocks();
			size_t maxThreads = 256; // this->m_selected_spec->GPUThreads();
			const size_t overlap = this->m_overlap;
			const size_t numrows = arg.total_rows();
			const size_t rowWidth = arg.total_cols();
			size_t trdsize = rowWidth;
			size_t blocksPerRow = 1;
			
			if (rowWidth > maxThreads)
			{
				size_t tmp = getThreadNumber_CU<T>(rowWidth, maxThreads, deviceID);
				if (tmp < 1 || tmp * numrows > maxBlocks)
					SKEPU_ERROR("CUDA MapOverlap: Row width is larger than maximum thread size: " << rowWidth << " " << maxThreads);
				blocksPerRow = tmp;
				trdsize = rowWidth / blocksPerRow;
				
				if (trdsize < overlap)
					SKEPU_ERROR("CUDA MapOverlap: Cannot execute overlap with current overlap width");
			}
			
			// Allocate wrap vector to device.
			std::vector<T> wrap(2 * overlap * numrows);
			DeviceMemPointer_CU<T> wrap_mem_p(&wrap[0], wrap.size(), this->m_environment->m_devices_CU[deviceID]);
			
			if (this->m_edge == Edge::Cyclic)
			{
				// Just update here to get latest values back.
				arg.updateHost();
				auto inputBeginTemp = arg.begin();
				
				for (size_t row = 0; row < numrows; row++)
				{
					auto inputEndTemp = inputBeginTemp+rowWidth;
					
					for (size_t i = 0; i < overlap; ++i)
					{
						wrap[i + row * 2 * overlap] = inputEndTemp(i - overlap);// inputEndMinusOverlap(i);
						wrap[overlap + i + row * 2 * overlap] = inputBeginTemp(i);
					}
					inputBeginTemp += rowWidth;
				}
				// Copy wrap vector only if it is a CYCLIC overlap policy.
				wrap_mem_p.copyHostToDevice();
			}
			
			const size_t numThreads = trdsize; //std::min(maxThreads, rowWidth);
			const size_t numBlocks = std::max<size_t>(1, std::min(blocksPerRow * numrows, maxBlocks));
			const size_t sharedMemSize = sizeof(T) * (numThreads + 2 * overlap);
			
			// Copy elements to device and allocate output memory.
			auto in_mem_p = arg.updateDevice_CU(arg.getAddress(), n, deviceID, AccessMode::Read, false);
			auto out_mem_p = res.updateDevice_CU(res.getAddress(), n, deviceID, AccessMode::Write, false);
			auto anyMemP = std::make_tuple(get<AI, CallArgs...>(args...).cudaProxy(deviceID, MapOverlapFunc::anyAccessMode[AI])...);
			
#ifdef USE_PINNED_MEMORY
			this->m_cuda_rowwise_kernel<<<numBlocks, numThreads, sharedMemSize, this->m_environment->m_devices_CU.at(deviceID)->m_streams[0]>>>
#else
			this->m_cuda_rowwise_kernel<<<numBlocks, numThreads, sharedMemSize>>>
#endif
			(
				in_mem_p->getDeviceDataPointer(),
				std::get<AI>(anyMemP).second...,
				get<CI, CallArgs...>(args...)...,
				out_mem_p->getDeviceDataPointer(),
				wrap_mem_p.getDeviceDataPointer(),
				n, 0, n,
				static_cast<int>(this->m_edge),
				this->m_pad, this->m_overlap,
				blocksPerRow, rowWidth
			);
			
			// Make sure the data is marked as changed by the device
			out_mem_p->changeDeviceData();
		}
		
		
		/*!
		*  Performs the row-wise MapOverlap on a range of elements, using multiple GPUs, on the \em CUDA with a seperate output range.
		*  Used internally by other methods to apply row-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::mapOverlapMultiThread_CU_Row(size_t numDevices, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			const size_t totalElems = arg.size();
			const size_t maxBlocks = this->m_selected_spec->GPUBlocks();
			size_t maxThreads = this->m_selected_spec->GPUThreads();
			const size_t overlap = this->m_overlap;
			const size_t numrows = arg.total_rows();
			const size_t rowWidth = arg.total_cols();
			size_t trdsize= rowWidth;
			size_t blocksPerRow = 1;
			
			if (rowWidth > maxThreads)
			{
				size_t tmp = getThreadNumber_CU<T>(rowWidth, maxThreads, 0);
				if (tmp < 1 || tmp * numrows > maxBlocks)
					SKEPU_ERROR("CUDA MapOverlap: Row width is larger than maximum thread size: " << rowWidth << " " << maxThreads);
				blocksPerRow = tmp;
				trdsize = rowWidth / blocksPerRow;
				
				if (trdsize < overlap)
					SKEPU_ERROR("CUDA MapOverlap: Cannot execute overlap with current overlap width");
			}
			
			const size_t numRowsPerSlice = numrows / numDevices;
			const size_t numElemPerSlice = numRowsPerSlice * rowWidth;
			const size_t restRows = numrows % numDevices;
			
			//Need to get new values from other devices so that the overlap between devices is up to date.
			//Bad for performance since whole vector needs to be copied, fix so that only overlap is fetched and updated.
		//	arg.updateHostAndInvalidateDevice();
			
			std::vector<T> wrap(2 * overlap * numrows);
			
			if (this->m_edge == Edge::Cyclic)
			{
				// Just update here to get latest values back.
				arg.updateHost();
				auto inputBeginTemp = arg.begin();
			
				for (size_t row=0; row< numrows; row++)
				{
					auto inputEndTemp = inputBeginTemp + rowWidth;
					
					for (size_t i = 0; i < overlap; ++i)
					{
						wrap[i+(row * 2 * overlap)] = inputEndTemp(i - overlap);// inputEndMinusOverlap(i);
						wrap[(overlap+i)+(row * 2 * overlap)] = inputBeginTemp(i);
					}
					inputBeginTemp += rowWidth;
				}
			}
			
			size_t numThreads = trdsize; //std::min(maxThreads, rowWidth);
			
			DeviceMemPointer_CU<T> *in_mem_p[MAX_GPU_DEVICES];
			DeviceMemPointer_CU<Ret> *out_mem_p[MAX_GPU_DEVICES];
			DeviceMemPointer_CU<T> *wrap_mem_p[MAX_GPU_DEVICES];
			
			// First create CUDA memory if not created already.
			for (size_t i = 0; i < numDevices; ++i)
			{
				const size_t numRows = numRowsPerSlice + ((i == numDevices - 1) ? restRows : 0);
				
				// Copy wrap vector to device.
				wrap_mem_p[i] = new DeviceMemPointer_CU<T>(&wrap[i * numRowsPerSlice * overlap * 2], numRowsPerSlice * overlap * 2, this->m_environment->m_devices_CU.at(i));
				
				// Copy elements to device and allocate output memory.
				in_mem_p[i]  = arg.updateDevice_CU(arg.getAddress() + i * numElemPerSlice, numRows * rowWidth, i, AccessMode::None, false);
				out_mem_p[i] = res.updateDevice_CU(res.getAddress() + i * numElemPerSlice, numRows * rowWidth, i, AccessMode::None, false);
			}
			
			// we will divide the computation row-wise... copy data and do operation
			for (size_t i = 0; i < numDevices; ++i)
			{
				const size_t numRows = numRowsPerSlice + ((i == numDevices - 1) ? restRows : 0);
				const size_t numElems = numRows * rowWidth;
				
				// Copy wrap vector only if it is a CYCLIC overlap policy.
				if (this->m_edge == Edge::Cyclic)
					wrap_mem_p[i]->copyHostToDevice();
				
				const size_t numBlocks = std::max<size_t>(1, std::min(blocksPerRow * numRows, maxBlocks));
				const size_t sharedMemSize = sizeof(T) * (numThreads + 2 * overlap);
				
				// Copy elements to device and allocate output memory.
				in_mem_p[i]  = arg.updateDevice_CU(arg.getAddress() + i * numElemPerSlice, numRows * rowWidth, i, AccessMode::Read, false);
				out_mem_p[i] = res.updateDevice_CU(res.getAddress() + i * numElemPerSlice, numRows * rowWidth, i, AccessMode::Write, false);
				
				auto anyMemP = std::make_tuple(get<AI, CallArgs...>(args...).cudaProxy(i, MapOverlapFunc::anyAccessMode[AI])...);
				
				cudaSetDevice(i);
				
#ifdef USE_PINNED_MEMORY
				this->m_cuda_rowwise_kernel<<<numBlocks, numThreads, sharedMemSize, this->m_environment->m_devices_CU.at(i)->m_streams[0]>>>
#else
				this->m_cuda_rowwise_kernel<<<numBlocks, numThreads, sharedMemSize>>>
#endif
				(
					in_mem_p[i]->getDeviceDataPointer(),
					std::get<AI>(anyMemP).second...,
					get<CI, CallArgs...>(args...)...,
					out_mem_p[i]->getDeviceDataPointer(),
					wrap_mem_p[i]->getDeviceDataPointer(),
					numElems, 0, numElems,
					static_cast<int>(this->m_edge),
					this->m_pad, this->m_overlap,
					blocksPerRow, rowWidth
				);
				
				// Make sure the data is marked as changed by the device
				out_mem_p[i]->changeDeviceData();
			}
			
			// to properly de-allocate the memory
			for (size_t i = 0; i < numDevices; ++i)
				delete wrap_mem_p[i];
			
			cudaSetDevice(this->m_environment->bestCUDADevID);
		}
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::rowwise_CUDA(Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("CUDA MapOverlap 1D Matrix: size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CU.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			if (numDevices <= 1)
				return this->mapOverlapSingleThread_CU_Row(0, res, arg, ai, ci, args...);
			else
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
				return this->mapOverlapMultiThread_CU_Row(numDevices, res, arg, ai, ci, args...);
		}
		
		
		
		
		
		
		/*!
		 *  Performs the 2D MapOverlap using a single CUDA GPU.
		 *  The actual filter is specified in a user-function.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>
		::mapOverlapSingleThread_CU(size_t deviceID, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...>, pack_indices<CI...>,  CallArgs&&... args)
		{
			const size_t in_rows = arg.total_rows();
			const size_t in_cols = arg.total_cols();
			const size_t out_rows = res.total_rows();
			const size_t out_cols = res.total_cols();
			
			cudaSetDevice(deviceID);
			
			typename Matrix<T>::device_pointer_type_cu in_mem_p = arg.updateDevice_CU(arg.GetArrayRep(), in_rows * in_cols, deviceID, AccessMode::Read, true);
			typename Matrix<T>::device_pointer_type_cu out_mem_p = res.updateDevice_CU(res.GetArrayRep(), out_rows * out_cols, deviceID, AccessMode::Write, true);
			
			dim3 numBlocks, numThreads;
			
			numThreads.x = (out_cols > 16) ? 16 : out_cols;
			numThreads.y = (out_rows > 32) ? 32 : out_rows;
			numThreads.z = 1;
			
			numBlocks.x = (out_cols + numThreads.x - 1) / numThreads.x;
			numBlocks.y = (out_rows + numThreads.y - 1) / numThreads.y;
			numBlocks.z = 1;
			
			auto anyMemP = std::make_tuple(get<AI, CallArgs...>(args...).cudaProxy(deviceID, MapOverlapFunc::anyAccessMode[AI])...);
			
			size_t sharedMem =  (numThreads.x + this->m_overlap_x * 2) * (numThreads.y + this->m_overlap_y * 2) * sizeof(T);
			
#ifdef USE_PINNED_MEMORY
			this->m_cuda_kernel<<<numBlocks,numThreads, sharedMem, this->m_environment->m_devices_CU.at(deviceID)->m_streams[0]>>>
#else
			this->m_cuda_kernel<<<numBlocks,numThreads, sharedMem>>>
#endif
			(
				in_mem_p->getDeviceDataPointer(),
				std::get<AI>(anyMemP).second...,
				get<CI, CallArgs...>(args...)...,
				out_mem_p->getDeviceDataPointer(),
				out_rows, out_cols,
				this->m_overlap_y, this->m_overlap_x,
				in_cols, out_cols,
				numThreads.y + this->m_overlap_y * 2,
				numThreads.x + this->m_overlap_x * 2
			);
			
			// Make sure the data is marked as changed by the device
			out_mem_p->changeDeviceData();
		}
		
		
		/*!
		*  Performs the 2D MapOverlap using multiple CUDA GPUs.
		*  The actual filter is specified in a user-function.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>
		::mapOverlapMultipleThread_CU(size_t numDevices, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...>, pack_indices<CI...>,  CallArgs&&... args)
		{
			const size_t in_rows = arg.total_rows();
			const size_t in_cols = arg.total_cols();
			const size_t out_rows = res.total_rows();
			const size_t out_cols = res.total_cols();
			
			const size_t numRowsPerSlice = out_rows / numDevices;
			const size_t restRows = out_rows % numDevices;
			
			// Need to get new values from other devices so that the overlap between devices is up to date.
			// Bad for performance since whole vector needs to be copied, fix so that only overlap is fetched and updated.
		//	input.updateHostAndInvalidateDevice();
			
			typename Matrix<T>::device_pointer_type_cu in_mem_p[MAX_GPU_DEVICES];
			typename Matrix<T>::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];
			
			// First create CUDA memory if not created already.
			for (size_t i = 0; i < numDevices; ++i)
			{
				cudaSetDevice(i);
				
				size_t outRows;
				if (i == numDevices-1)
					outRows = numRowsPerSlice+restRows;
				else
					outRows = numRowsPerSlice;
				
				size_t inRows = outRows + this->m_overlap_y * 2; // no matter which device, number of input rows is same.
				
				in_mem_p[i] = arg.updateDevice_CU(arg.GetArrayRep() + i * numRowsPerSlice * in_cols, inRows * in_cols, i, AccessMode::None, true);
				out_mem_p[i] = res.updateDevice_CU(res.GetArrayRep() + i * numRowsPerSlice * out_cols, outRows * out_cols, i, AccessMode::None, true);
			}
			
			// Fill out argument struct with right information and start threads.
			for (size_t i = 0; i < numDevices; ++i)
			{
				cudaSetDevice(i);
				
				size_t outRows;
				if (i == numDevices-1)
					outRows = numRowsPerSlice+restRows;
				else
					outRows = numRowsPerSlice;
				
				size_t inRows = outRows + this->m_overlap_y * 2; // no matter which device, number of input rows is same.
				
				in_mem_p[i] = arg.updateDevice_CU(arg.GetArrayRep() + i * numRowsPerSlice * in_cols, inRows * in_cols, i, AccessMode::Read, true);
				out_mem_p[i] = res.updateDevice_CU(res.GetArrayRep() + i * numRowsPerSlice * out_cols, outRows * out_cols, i, AccessMode::Write, true, true);
				
				dim3 numBlocks;
				dim3 numThreads;
				
				numThreads.x = (out_cols > 16) ? 16 : out_cols;
				numThreads.y = (outRows > 32) ? 32 : outRows;
				numThreads.z = 1;
				
				numBlocks.x = (out_cols + numThreads.x - 1) / numThreads.x;
				numBlocks.y = (outRows + numThreads.y - 1) / numThreads.y;
				numBlocks.z = 1;
				
				size_t sharedMem =  (numThreads.x + this->m_overlap_x * 2) * (numThreads.y + this->m_overlap_y * 2) * sizeof(T);
				
				auto anyMemP = std::make_tuple(get<AI, CallArgs...>(args...).cudaProxy(i, MapOverlapFunc::anyAccessMode[AI])...);
				
#ifdef USE_PINNED_MEMORY
				this->m_cuda_kernel<<< numBlocks,numThreads, sharedMem, this->m_environment->m_devices_CU[i]->m_streams[0]>>>
#else
				this->m_cuda_kernel<<< numBlocks,numThreads, sharedMem >>>
#endif
				(
					in_mem_p[i]->getDeviceDataPointer(), 
					std::get<AI>(anyMemP).second...,
					get<CI, CallArgs...>(args...)...,
					out_mem_p[i]->getDeviceDataPointer(),
					outRows, out_cols,
					this->m_overlap_y, this->m_overlap_x,
					in_cols, out_cols,
					numThreads.y + this->m_overlap_y * 2,
					numThreads.x + this->m_overlap_x * 2
				);
				
				// Make sure the data is marked as changed by the device
				out_mem_p[i]->changeDeviceData();
			}
		}
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_CUDA(skepu2::Matrix<Ret>& res, skepu2::Matrix<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci,  CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("CUDA MapOverlap 2D: size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CU.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			if (numDevices <= 1)
				return this->mapOverlapSingleThread_CU(0, res, arg, ai, ci, get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
			else
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
				return this->mapOverlapMultipleThread_CU(numDevices, res, arg, ai, ci, get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
		}
		
	}
}

#endif
