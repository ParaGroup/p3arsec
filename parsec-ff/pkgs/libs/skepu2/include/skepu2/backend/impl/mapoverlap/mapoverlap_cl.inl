/*! \file mapoverlap_cl.inl
*  \brief Contains the definitions of OpenCL specific member functions for the MapOverlap skeleton.
 */

#ifdef SKEPU_OPENCL

namespace skepu2
{
	namespace backend
	{
		/*!
		 *  Applies the MapOverlap skeleton to a range of elements specified by iterators. Result is saved to a seperate output range.
		 *  The function uses only \em one device which is decided by a parameter. Using \em OpenCL as backend.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<template<class> class Container, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::mapOverlapSingle_CL(size_t deviceID, Container<Ret>& res, Container<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			// Setup parameters
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			size_t n = arg.size();
			size_t overlap = this->m_overlap;
			size_t out_offset = 0;
			size_t out_numelements = n;
			
			// Sets the pad and edge policy values that are sent to the kernel
			const int _poly = static_cast<int>(this->m_edge);
			const T _pad = (this->m_edge == Edge::Pad) ? this->m_pad : 0;
			
			// Constructs a wrap vector, which is used if cyclic edge policy is specified.
			std::vector<T> wrap(2 * overlap);
			if (this->m_edge == Edge::Cyclic)
			{
				arg.updateHostAndInvalidateDevice();
				for (size_t i = 0; i < overlap; ++i)
				{
					wrap[i] = arg.end()(i - overlap);
					wrap[overlap+i] = arg(i);
				}
			}
			
			// Copy wrap vector to device.
			DeviceMemPointer_CL<T> wrap_mem_p(&wrap[0], wrap.size(), device);
			wrap_mem_p.copyHostToDevice();
			
			const size_t numThreads = std::min<size_t>(this->m_selected_spec->GPUThreads(), n);
			const size_t numBlocks = std::max<size_t>(1, std::min(n / numThreads + (n % numThreads == 0 ? 0 : 1), this->m_selected_spec->GPUBlocks()));
			const size_t sharedMemSize = sizeof(T) * (numThreads + 2 * overlap);
			
			// Copy elements to device and allocate output memory.
			auto in_mem_p = arg.updateDevice_CL(arg.getAddress(), n, device, true);
			auto out_mem_p = res.updateDevice_CL(res.getAddress(), n, device, false);
			
			auto anyMemP = std::make_tuple(get<AI, CallArgs...>(args...).getParent().updateDevice_CL(get<AI, CallArgs...>(args...)
					.getAddress(), get<AI, CallArgs...>(args...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI]))...);
			
			CLKernel::mapOverlapVector(
				deviceID, numThreads, numBlocks * numThreads,
				in_mem_p,
				std::make_tuple(&get<AI, CallArgs...>(args...).getParent(), std::get<AI>(anyMemP))...,
				get<CI, CallArgs...>(args...)...,
				out_mem_p, &wrap_mem_p,
				n, overlap, out_offset, out_numelements, _poly, _pad,
				sharedMemSize
			);
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<AI>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			out_mem_p->changeDeviceData();
		}

		/*!
		 *  Applies the MapOverlap skeleton to a range of elements specified by iterators. Result is saved to a seperate output range.
		 *  The function uses a variable number of devices, dividing the range of elemets equally among the participating devices each mapping
		 *  its part. Using \em OpenCL as backend.
		 */
		template <typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<template<class> class Container, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::mapOverlapNumDevices_CL(size_t numDevices, Container<Ret>& res, Container<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			// Divide the elements amongst the devices
			const size_t size = arg.size();
			const size_t numElemPerSlice = size / numDevices;
			const size_t rest = size % numDevices;
			const size_t overlap = this->m_overlap;
			
			// Sets the pad and edge policy values that are sent to the kernel
			const int _poly = static_cast<int>(this->m_edge);
			const T _pad = (this->m_edge == Edge::Pad) ? this->m_pad : 0;
			
			// Constructs a wrap vector, which is used if cyclic edge policy is specified.
			std::vector<T> wrap(2 * overlap);
			
			if (this->m_edge == Edge::Cyclic)
			{
				// Just update here to get latest values back.
				arg.updateHost();
				for (size_t i = 0; i < overlap; ++i)
				{
					wrap[i] = arg.end()(i - overlap);
					wrap[overlap + i] = arg(i);
				}
			}
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numElem = numElemPerSlice + ((i == numDevices - 1) ? rest : 0);
				size_t out_offset, n;
				
				// Copy wrap vector to device.
				DeviceMemPointer_CL<T> wrap_mem_p(&wrap[0], wrap.size(), device);
				wrap_mem_p.copyHostToDevice();
				
				// Copy elemets to device and set other kernel parameters depending on which device it is, first, last or a middle device.
				if (i == 0)                 { out_offset = 0;       n = numElem + overlap; }
				else if (i == numDevices-1) { out_offset = overlap; n = numElem + overlap; }
				else                        { out_offset = overlap; n = numElem + overlap * 2; }
				
				size_t out_numelements = numElem;
				auto in_mem_p = arg.updateDevice_CL(arg.getAddress() + i * numElemPerSlice - out_offset, n, device, true);
				
				// Setup parameters
				const size_t numThreads = std::min<size_t>(this->m_selected_spec->GPUThreads(), n);
				const size_t numBlocks = std::max<size_t>(1, std::min(n / numThreads + (n % numThreads == 0 ? 0:1), this->m_selected_spec->GPUBlocks()));
				const size_t sharedMemSize = sizeof(T) * (numThreads + 2 * overlap);
				
				// Allocate memory for output.
				auto out_mem_p = res.updateDevice_CL(res.getAddress() + i * numElemPerSlice, numElem, device, false);
				auto anyMemP = std::make_tuple(get<AI, CallArgs...>(args...).getParent().updateDevice_CL(get<AI, CallArgs...>(args...)
					.getAddress(), get<AI, CallArgs...>(args...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI]))...);
				
				CLKernel::mapOverlapVector(
					i, numThreads, numBlocks * numThreads,
					in_mem_p,
					std::make_tuple(&get<AI, CallArgs...>(args...).getParent(), std::get<AI>(anyMemP))...,
					get<CI, CallArgs...>(args...)...,
					out_mem_p, &wrap_mem_p,
					n, overlap, out_offset, out_numelements, _poly, _pad,
					sharedMemSize
				);
				
				// Make sure the data is marked as changed by the device
				pack_expand((std::get<AI>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
				out_mem_p->changeDeviceData();
			}
		}
		
		
		/*!
		 *  Performs the MapOverlap on a range of elements. With a seperate output range. The function decides whether to perform
		 *  the MapOverlap on one device, calling mapOverlapSingle_CL or
		 *  on multiple devices, calling mapOverlapNumDevices_CL.
		 *  Using \em OpenCL as backend.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<template<class> class Container, size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::vector_OpenCL(Container<Ret>& res, Container<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenCL MapOverlap 1D Matrix: size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			if (numDevices <= 1)
				return this->mapOverlapSingle_CL(0, res, arg, ai, ci, args...);
			else
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
				return this->mapOverlapNumDevices_CL(numDevices, res, arg, ai, ci, args...);
		}
		
		
		/*!
		 * For Matrix overlap, we need to check whether overlap configuration is runnable considering total size of shared memory available on that system.
		 * This method is a helper funtion doing that. It is called by another helper \p getThreadNumber_CL() method.
		 *
		 * \param numThreads Number of threads in a thread block.
		 * \param deviceID The device ID.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<typename T>
		bool MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::sharedMemAvailable_CL(size_t &numThreads, size_t deviceID)
		{
			size_t overlap = this->m_overlap;
			size_t maxShMem = this->m_environment->m_devices_CL.at(deviceID)->getSharedMemPerBlock() / sizeof(T) - SHMEM_SAFITY_BUFFER; // little buffer for other usage
			size_t orgThreads = numThreads;
			
			numThreads = (numThreads + 2 * overlap < maxShMem) ? numThreads : maxShMem - 2 * overlap;
			
			if (orgThreads == numThreads) // return true when nothing changed because of overlap constraint
				return true;
			
			if (numThreads < 8) // not original numThreads then atleast 8 threads should be there
				SKEPU_ERROR("Possibly overlap is too high for operation to be successful on this GPU. MapOverlap Aborted");
				
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
		int MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::getThreadNumber_CL(size_t width, size_t numThreads, size_t deviceID)
		{
			// first check whether shared memory would be ok for this numThreads. Changes numThreads accordingly
			if (!sharedMemAvailable_CL<T>(numThreads, deviceID) && numThreads < 1)
				SKEPU_ERROR("Too low overlap size to continue.");
			
			if (width % numThreads == 0)
				return width / numThreads;
			
			for (size_t i = numThreads - 1; i >= 1; i--) // decreament numThreads and see which one is a perfect divisor
			{
				if (width % numThreads == 0)
					return width / numThreads;
			}
			return -1;
		}
		
		
		/*!
		 *  Performs the row-wise MapOverlap on a range of elements on the \em OpenCL with a seperate output range.
		 *  Used internally by other methods to apply row-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::mapOverlapSingle_CL_Row(size_t deviceID, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			const size_t n = arg.size();
			const size_t overlap = this->m_overlap;
			const size_t out_offset = 0;
			const size_t out_numelements = n;
			const size_t numrows = arg.total_rows();
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			const size_t maxBlocks = this->m_selected_spec->GPUBlocks();
			size_t rowWidth = arg.total_cols(); // same as numcols
			size_t trdsize = rowWidth;
			size_t blocksPerRow = 1;
			
			if (rowWidth > this->m_selected_spec->GPUThreads())
			{
				int tmp = getThreadNumber_CL<T>(rowWidth, maxThreads, deviceID);
				if (tmp == -1 || tmp * numrows > maxBlocks)
					SKEPU_ERROR("Row width is larger than maximum thread size: " << rowWidth << " " << maxThreads);
				
				blocksPerRow = tmp;
				trdsize = rowWidth / blocksPerRow;
				
				if (trdsize < overlap)
					SKEPU_ERROR("Cannot execute overlap with current overlap width");
			}
			
			// Sets the pad and edge policy values that are sent to the kernel
			const int _poly = static_cast<int>(this->m_edge);
			const T _pad = (this->m_edge == Edge::Pad) ? this->m_pad : 0;
			
			// Constructs a wrap vector, which is used if cyclic edge policy is specified.
			std::vector<T> wrap(2 * overlap * numrows);
			
			if (this->m_edge == Edge::Cyclic)
			{
				// Just update here to get latest values back.
				arg.updateHost();
				auto inputBeginTemp = arg.begin();
				
				for (size_t row = 0; row < numrows; row++)
				{
					auto inputEndTemp = inputBeginTemp + rowWidth;
					for (size_t i = 0; i < overlap; ++i)
					{
						wrap[i + row * overlap * 2] = inputEndTemp(i - overlap);// inputEndMinusOverlap(i);
						wrap[i + row * overlap * 2 + overlap] = inputBeginTemp(i);
					}
					inputBeginTemp += rowWidth;
				}
			}
			/*    else
         wrap.resize(1); // not used so minimize overhead;*/
			
			// Copy wrap vector to device.
			DeviceMemPointer_CL<T> wrapMemP(&wrap[0], wrap.size(), device);
			wrapMemP.copyHostToDevice();
			
			const size_t numThreads = trdsize; // std::min<size_t>(this->m_selected_spec->GPUThreads(), n);
			const size_t numBlocks = std::max<size_t>(1, std::min(blocksPerRow * numrows, this->m_selected_spec->GPUBlocks()));
			const size_t sharedMemSize = sizeof(T) * (numThreads+2*overlap);
			
			// Copy elements to device and allocate output memory.
			auto inMemP  = arg.updateDevice_CL(arg.getAddress(), numrows, rowWidth, device, true);
			auto outMemP = res.updateDevice_CL(res.getAddress(), numrows, rowWidth, device, false);
			auto anyMemP = std::make_tuple(get<AI, CallArgs...>(args...).getParent().updateDevice_CL(get<AI, CallArgs...>(args...).getAddress(),
				get<AI, CallArgs...>(args...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI]))...);
			
			CLKernel::mapOverlapMatrixRowWise(
				deviceID, numThreads, numBlocks * numThreads,
				inMemP,
				std::make_tuple(&get<AI, CallArgs...>(args...).getParent(), std::get<AI>(anyMemP))...,
				get<CI, CallArgs...>(args...)...,
				outMemP, &wrapMemP, n,
				overlap, out_offset, out_numelements, _poly, _pad, blocksPerRow, rowWidth,
				sharedMemSize
			);
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<AI>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			outMemP->changeDeviceData();
		}
		
		
		/*!
		 *  Performs the row-wise MapOverlap on a range of elements on the \em OpenCL with a seperate output range.
		 *  Used internally by other methods to apply row-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::mapOverlapSingle_CL_RowMulti(size_t numDevices, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			const size_t totalElems = arg.size();
			const size_t overlap = this->m_overlap;
			const size_t numrows = arg.total_rows();
			size_t rowWidth = arg.total_cols();
			size_t trdsize = rowWidth;
			size_t blocksPerRow = 1;
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			const size_t maxBlocks = this->m_selected_spec->GPUBlocks();
			
			if (rowWidth > maxThreads)
			{
				int tmp = getThreadNumber_CL<T>(rowWidth, maxThreads, 0);
				if (tmp == -1 || tmp * numrows > maxBlocks)
					SKEPU_ERROR("ERROR! Row width is larger than maximum thread size: " << rowWidth << " " << maxThreads);
				
				blocksPerRow = tmp;
				trdsize = rowWidth / blocksPerRow;
				
				if (trdsize < overlap)
					SKEPU_ERROR("ERROR! Cannot execute overlap with current overlap width.");
			}
			
			// Sets the pad and edge policy values that are sent to the kernel
			const int _poly = static_cast<int>(this->m_edge);
			const T _pad = (this->m_edge == Edge::Pad) ? this->m_pad : 0;
			
			const size_t numRowsPerSlice = numrows / numDevices;
			const size_t numElemPerSlice = numRowsPerSlice * rowWidth;
			const size_t restRows = numrows % numDevices;
			
			//Need to get new values from other devices so that the overlap between devices is up to date.
			//Bad for performance since whole vector needs to be copied, fix so that only overlap is fetched and updated.
		//	inputBegin.getParent().updateHostAndInvalidateDevice();
			
			std::vector<T> wrap(overlap * 2 * numrows);
			
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
						wrap[i + row * overlap * 2] = inputEndTemp(i - overlap);// inputEndMinusOverlap(i);
						wrap[overlap + i + row * overlap * 2] = inputBeginTemp(i);
					}
					inputBeginTemp += rowWidth;
				}
			}
			
			DeviceMemPointer_CL<T> *in_mem_p[MAX_GPU_DEVICES], *out_mem_p[MAX_GPU_DEVICES], *wrap_mem_p[MAX_GPU_DEVICES];
			
			// First create OpenCL memory if not created already.
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numRows = numRowsPerSlice + ((i == numDevices - 1) ? restRows : 0);
				
				// Copy wrap vector to device.
				wrap_mem_p[i] = new DeviceMemPointer_CL<T>(&wrap[i * numRowsPerSlice * overlap * 2], &wrap[i * numRowsPerSlice * overlap * 2], numRowsPerSlice * overlap * 2, device);
				
				// Copy elements to device and allocate output memory.
				in_mem_p[i]  = arg.updateDevice_CL(arg.getAddress() + i * numElemPerSlice, numRows, rowWidth, device, false);
				out_mem_p[i] = arg.updateDevice_CL(res.getAddress() + i * numElemPerSlice, numRows, rowWidth, device, false);
			}
			
			size_t out_offset = 0;
			
			// we will divide the computation row-wise... copy data and do operation
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numRows = numRowsPerSlice + ((i == numDevices - 1) ? restRows : 0);
				const size_t numElems = numRows * rowWidth;
				
				// Copy wrap vector only if it is a CYCLIC overlap policy.
				if (this->m_edge == Edge::Cyclic)
					wrap_mem_p[i]->copyHostToDevice();
				
				const size_t numThreads = trdsize; //std::min(maxThreads, rowWidth);
				const size_t numBlocks = std::max<size_t>(1, std::min(blocksPerRow * numRows, maxBlocks));
				const size_t sharedMemSize = sizeof(T) * (numThreads + 2 * overlap);
				
				// Copy elements to device and allocate output memory.
				in_mem_p[i] = arg.updateDevice_CL(arg.getAddress() + i * numElemPerSlice, numRows, rowWidth, device, true);
				
				auto anyMemP = std::make_tuple(get<AI, CallArgs...>(args...).getParent().updateDevice_CL(get<AI, CallArgs...>(args...).getAddress(),
					get<AI, CallArgs...>(args...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI]))...);
				
				CLKernel::mapOverlapMatrixRowWise(
					i, numThreads, numBlocks * numThreads,
					in_mem_p[i],
					std::make_tuple(&get<AI, CallArgs...>(args...).getParent(), std::get<AI>(anyMemP))...,
					get<CI, CallArgs...>(args...)...,
					out_mem_p[i], wrap_mem_p[i],
					numElems, overlap, out_offset, numElems, _poly, _pad,
					blocksPerRow, rowWidth,
					sharedMemSize
				);
				
				// Make sure the data is marked as changed by the device
				pack_expand((std::get<AI>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
				out_mem_p[i]->changeDeviceData();
			}
			
			// to properly de-allocate the memory
			for (size_t i = 0; i < numDevices; ++i)
				delete wrap_mem_p[i];
		}
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::rowwise_OpenCL(Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenCL MapOverlap 1D Matrix: size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			if (numDevices <= 1)
				return this->mapOverlapSingle_CL_Row(0, res, arg, ai, ci, args...);
			else
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
				return this->mapOverlapSingle_CL_RowMulti(numDevices, res, arg, ai, ci, args...);
		}
		
		
		/*!
		 *  Performs the column-wise MapOverlap on a range of elements on the \em OpenCL with a seperate output range.
		 *  Used internally by other methods to apply row-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::mapOverlapSingle_CL_Col(size_t deviceID, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			const size_t n = arg.size();
			const size_t overlap = this->m_overlap;
			const size_t out_offset = 0;
			const size_t out_numelements = n;
			const size_t colWidth = arg.total_rows();
			const size_t numcols = arg.total_cols();
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			const size_t maxBlocks = this->m_selected_spec->GPUBlocks();
			size_t trdsize = colWidth;
			size_t blocksPerCol = 1;
			
			if (colWidth > maxThreads)
			{
				int tmp = getThreadNumber_CL<T>(colWidth, maxThreads, deviceID);
				if (tmp == -1 || blocksPerCol * numcols > maxBlocks)
					SKEPU_ERROR("Col width is larger than maximum thread size: " << colWidth << " " << maxThreads);
				
				blocksPerCol = tmp;
				trdsize = colWidth / blocksPerCol;
				if (trdsize < overlap)
					SKEPU_ERROR("Thread size should be larger than overlap width");
			}
			
			// Sets the pad and edge policy values that are sent to the kernel
			const int _poly = static_cast<int>(this->m_edge);
			const T _pad = (this->m_edge == Edge::Pad) ? this->m_pad : 0;
			
			// Constructs a wrap vector, which is used if cyclic edge policy is specified.
			std::vector<T> wrap(overlap * 2 * (n / colWidth));
			
			if (this->m_edge == Edge::Cyclic)
			{
				arg.updateHost();
				auto inputBeginTemp = arg.begin();
				
				for (size_t col=0; col< numcols; col++)
				{
					auto inputEndTemp = inputBeginTemp + numcols * (colWidth - 1);
					auto inputEndMinusOverlap = (inputEndTemp - (overlap - 1) * numcols);
					
					for (size_t i = 0; i < overlap; ++i)
					{
						wrap[i + col * overlap * 2] = inputEndMinusOverlap(i * numcols);
						wrap[overlap + i + col * overlap * 2] = inputBeginTemp(i * numcols);
					}
					inputBeginTemp++;
				}
			}
			// Copy wrap vector to device.
			DeviceMemPointer_CL<T> wrapMemP(&wrap[0], wrap.size(), device);
			wrapMemP.copyHostToDevice();
			
			const size_t numThreads = trdsize; //std::min(maxThreads, rowWidth);
			const size_t numBlocks = std::max<size_t>(1, std::min(blocksPerCol * numcols, maxBlocks));
			const size_t sharedMemSize = sizeof(T) * (numThreads+2*overlap);
			
			// Copy elements to device and allocate output memory.
			auto inMemP  = arg.updateDevice_CL(arg.getAddress(), colWidth, numcols, device, true);
			auto outMemP = res.updateDevice_CL(res.getAddress(), colWidth, numcols, device, false);
			auto anyMemP = std::make_tuple(get<AI, CallArgs...>(args...).getParent().updateDevice_CL(get<AI, CallArgs...>(args...).getAddress(),
				get<AI, CallArgs...>(args...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI]))...);
			
			CLKernel::mapOverlapMatrixColWise(
				deviceID, numThreads, numBlocks * numThreads,
				inMemP,
				std::make_tuple(&get<AI, CallArgs...>(args...).getParent(), std::get<AI>(anyMemP))...,
				get<CI, CallArgs...>(args...)...,
				outMemP, &wrapMemP,
				n, overlap, out_offset, out_numelements, _poly, _pad,
				blocksPerCol, numcols, colWidth,
				sharedMemSize
			);
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<AI>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			outMemP->changeDeviceData();
		}
		
		
		/*!
		*  Performs the column-wise MapOverlap on a range of elements on the \em OpenCL with a seperate output range.
		*  Used internally by other methods to apply column-wise mapoverlap operation.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::mapOverlapSingle_CL_ColMulti(size_t numDevices, Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			const size_t totalElems = arg.size();
			const size_t overlap = this->m_overlap;
			const size_t colWidth = arg.total_rows();
			const size_t numCols = arg.total_cols();
			const size_t numRowsPerSlice = colWidth / numDevices;
			const size_t numElemPerSlice = numRowsPerSlice * numCols;
			const size_t restRows = colWidth % numDevices;
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			const size_t maxBlocks = this->m_selected_spec->GPUBlocks();
			
			std::vector<T> wrapStartDev(overlap * numCols), wrapEndDev(overlap * numCols);
			
			if (this->m_edge == Edge::Cyclic)
			{
				arg.updateHost();
				auto inputBeginTemp = arg.begin();
				
				for (size_t col = 0; col < numCols; col++)
				{
					auto inputEndTemp = inputBeginTemp + numCols * (colWidth - 1);
					auto inputEndMinusOverlap = inputEndTemp - (overlap - 1) * numCols;
					
					for (size_t i = 0; i < overlap; ++i)
					{
						wrapStartDev[i + col * overlap] = inputEndMinusOverlap(i * numCols);
						wrapEndDev[i + col * overlap] = inputBeginTemp(i * numCols);
					}
					inputBeginTemp++;
				}
			}
			
			DeviceMemPointer_CL<T> *in_mem_p[MAX_GPU_DEVICES], *out_mem_p[MAX_GPU_DEVICES];
			DeviceMemPointer_CL<T> *wrapStartDev_mem_p[MAX_GPU_DEVICES], *wrapEndDev_mem_p[MAX_GPU_DEVICES];
			
			size_t trdsize[MAX_GPU_DEVICES];
			size_t blocksPerCol[MAX_GPU_DEVICES];
			
			// First create OpenCL memory if not created already.
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numRows = numRowsPerSlice + ((i == numDevices - 1) ? restRows : 0);
				
				trdsize[i] = numRows;
				blocksPerCol[i] = 1;
				
				if (numRows > maxThreads)
				{
					int tmp = getThreadNumber_CL<T>(numRows, maxThreads, i);
					if (tmp == -1 || tmp * numCols > maxBlocks)
						SKEPU_ERROR("Col width is larger than maximum thread size: " << colWidth << " " << maxThreads);
					
					blocksPerCol[i] = tmp;
					trdsize[i] = numRows / blocksPerCol[i];
					
					if (trdsize[i] < overlap)
						SKEPU_ERROR("Cannot execute overlap with current overlap width.");
				}
				
				size_t overlapElems = overlap*numCols;
				
				if (i == 0)                 in_mem_p[i] = arg.updateDevice_CL(arg.getAddress() + i * numElemPerSlice,                numRows + overlap,     numCols, device, false);
				else if (i == numDevices-1) in_mem_p[i] = arg.updateDevice_CL(arg.getAddress() + i * numElemPerSlice - overlapElems, numRows + overlap,     numCols, device, false);
				else                        in_mem_p[i] = arg.updateDevice_CL(arg.getAddress() + i * numElemPerSlice - overlapElems, numRows + overlap * 2, numCols, device, false);
				
				out_mem_p[i] = res.updateDevice_CL(res.getAddress() + i * numElemPerSlice, numRows, numCols, device, false);
				
				// wrap vector, don't copy just allocates space on OpenCL device.
				wrapStartDev_mem_p[i] = new DeviceMemPointer_CL<T>(&wrapStartDev[0], &wrapStartDev[0], wrapStartDev.size(), device);
				wrapEndDev_mem_p[i] = new DeviceMemPointer_CL<T>(&wrapEndDev[0], &wrapEndDev[0], wrapEndDev.size(), device);
			}
			
			int _deviceType[MAX_GPU_DEVICES];
			
			// Sets the pad and edge policy values that are sent to the kernel
			const int _poly = static_cast<int>(this->m_edge);
			const T _pad = (this->m_edge == Edge::Pad) ? this->m_pad : 0;
			
			// we will divide the computation row-wise... copy data and do operation
			for (size_t i = 0; i < numDevices; i++)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numRows = numRowsPerSlice + ((i == numDevices - 1) ? restRows : 0);
				const size_t numElems = numRows * numCols;
				size_t in_offset, overlapElems = overlap * numCols;
				
				// Copy elemets to device and set other kernel parameters depending on which device it is, first, last or a middle device.
				if (i == 0)
				{
					_deviceType[i] = -1;
					
					if (this->m_edge == Edge::Cyclic)
						wrapStartDev_mem_p[i]->copyHostToDevice(); // copy wrap as it will be used partially, may optimize further by placing upper and lower overlap in separate buffers for muultiple devices?
					
					in_mem_p[i] = arg.updateDevice_CL(arg.getAddress() + i * numElemPerSlice, numRows + overlap, numCols, device, true);
					in_offset = 0;
				}
				else if (i == numDevices-1)
				{
					_deviceType[i] = 1;
					
					// copy wrap as it will be used partially, may optimize further by placing upper and lower overlap in separate buffers for multiple devices?
					if (this->m_edge == Edge::Cyclic)
						wrapEndDev_mem_p[i]->copyHostToDevice(); 
					
					in_mem_p[i] = arg.updateDevice_CL(arg.getAddress() + i * numElemPerSlice - overlapElems, numRows + overlap, numCols, device, true);
					in_offset = overlapElems;
				}
				else
				{
					_deviceType[i] = 0;
					in_mem_p[i] = arg.updateDevice_CL(arg.getAddress() + i * numElemPerSlice - overlapElems, numRows + 2 * overlap, numCols, device, true);
					in_offset = overlapElems;
				}
				
				const size_t numThreads = trdsize[i]; //std::min(maxThreads, rowWidth);
				const size_t numBlocks = std::max<size_t>(1, std::min(blocksPerCol[i] * numCols, maxBlocks));
				const size_t sharedMemSize = sizeof(T) * (numThreads + 2 * overlap);
				
				DeviceMemPointer_CL<T> *useWrapMemP = (this->m_edge == Edge::Cyclic && i == numDevices - 1)
					? wrapEndDev_mem_p[i] : wrapStartDev_mem_p[i]; 
				
				auto anyMemP = std::make_tuple(get<AI, CallArgs...>(args...).getParent().updateDevice_CL(get<AI, CallArgs...>(args...).getAddress(),
					get<AI, CallArgs...>(args...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI]))...);
				
				CLKernel::mapOverlapMatrixColWiseMulti(
					i, numThreads, numBlocks * numThreads,
					in_mem_p[i],
					std::make_tuple(&get<AI, CallArgs...>(args...).getParent(), std::get<AI>(anyMemP))...,
					get<CI, CallArgs...>(args...)...,
					out_mem_p[i], useWrapMemP,
					numElems, overlap, in_offset,
					numElems, _poly, _deviceType[i], _pad,
					blocksPerCol[i], numCols, numRows,
					sharedMemSize
				);
				
				// Make sure the data is marked as changed by the device
				pack_expand((std::get<AI>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
				out_mem_p[i]->changeDeviceData();
			}
			
			// Properly de-allocate the memory
			for (size_t i = 0; i < numDevices; ++i)
			{
				delete wrapStartDev_mem_p[i];
				delete wrapEndDev_mem_p[i];
			}
		}
		
		
		/*!
		 *  Performs the MapOverlap on a range of elements. With a seperate output range. The function decides whether to perform
		 *  the MapOverlap on one device, calling mapOverlapSingle_CL or
		 *  on multiple devices, calling mapOverlapNumDevices_CL.
		 *  Using \em OpenCL as backend.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename C2, typename C3, typename C4, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap1D<MapOverlapFunc, CUDAKernel, C2, C3, C4, CLKernel>
		::colwise_OpenCL(Matrix<Ret>& res, Matrix<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenCL MapOverlap 1D Matrix: size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->eevices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			if (numDevices <= 1)
				return this->mapOverlapSingle_CL_Col(0, res, arg, ai, ci, args...);
			else
			
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
				
				return this->mapOverlapSingle_CL_ColMulti(numDevices, res, arg, ai, ci, args...);
		}
		
		
		
		/*!
		 *  Performs the 2D MapOverlap using a single OpenCL GPU.
		 *  The actual filter is specified in a user-function.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>
		::mapOverlapSingleThread_CL(size_t deviceID, skepu2::Matrix<Ret>& res, skepu2::Matrix<T>& arg, pack_indices<AI...>, pack_indices<CI...>,  CallArgs&&... args)
		{
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			const size_t in_rows  = arg.total_rows();
			const size_t in_cols  = arg.total_cols();
			const size_t out_rows = res.total_rows();
			const size_t out_cols = res.total_cols();
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			
			auto inMemP  = arg.updateDevice_CL(arg.GetArrayRep(), in_rows,  in_cols,  device, true);
			auto outMemP = res.updateDevice_CL(res.GetArrayRep(), out_rows, out_cols, device, false);
			auto anyMemP = std::make_tuple(get<AI, CallArgs...>(args...).getParent().updateDevice_CL(get<AI, CallArgs...>(args...).getAddress(),
				get<AI, CallArgs...>(args...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI]))...);
			
			size_t numThreads[2], numBlocks[2];
			numThreads[0] = std::min<size_t>(out_cols, 16);
			numThreads[1] = std::min(out_rows, maxThreads / 16);
			numBlocks[0] = (size_t)((out_cols + numThreads[0] - 1) / numThreads[0]) * numThreads[0];
			numBlocks[1] = (size_t)((out_rows + numThreads[1] - 1) / numThreads[1]) * numThreads[1];
			
			const size_t sharedCols = numThreads[0] + this->m_overlap_x * 2;
			const size_t sharedRows = numThreads[1] + this->m_overlap_y * 2;
			const size_t sharedMemSize =  sharedRows * sharedCols * sizeof(T);
			
			CLKernel::mapOverlap2D(
				deviceID, numThreads, numBlocks,
				inMemP,
				std::make_tuple(&get<AI, CallArgs...>(args...).getParent(), std::get<AI>(anyMemP))...,
				get<CI, CallArgs...>(args...)...,
				outMemP, out_rows, out_cols,
				this->m_overlap_y, this->m_overlap_x,
				in_cols, sharedRows, sharedCols,
				sharedMemSize
			);
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<AI>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
			outMemP->changeDeviceData();
		}
		
		
		/*!
		 *  Performs the 2D MapOverlap using multiple OpenCL GPUs.
		 *  The actual filter is specified in a user-function.
		 */
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>
		::mapOverlapMultipleThread_CL(size_t numDevices, skepu2::Matrix<Ret>& res, skepu2::Matrix<T>& arg, pack_indices<AI...>, pack_indices<CI...>,  CallArgs&&... args)
		{
			const size_t in_rows  = arg.total_rows();
			const size_t in_cols  = arg.total_cols();
			const size_t out_rows = res.total_rows();
			const size_t out_cols = res.total_cols();
			const size_t numRowsPerSlice = out_rows / numDevices;
			const size_t restRows = out_rows % numDevices;
			const size_t maxThreads = this->m_selected_spec->GPUThreads();
			
			// Need to get new values from other devices so that the overlap between devices is up to date.
			// Bad for performance since whole vector needs to be copied, fix so that only overlap is fetched and updated.
			arg.updateHostAndInvalidateDevice();
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t outRows = numRowsPerSlice + ((i == numDevices - 1) ? restRows : 0);
				const size_t inRows = outRows + 2 * this->m_overlap_y; // no matter which device, number of input rows is same.
				
				typename Matrix<T>::device_pointer_type_cl in_mem_p = arg.updateDevice_CL(arg.GetArrayRep() + i * numRowsPerSlice * in_cols, inRows, in_cols, device, true);
				typename Matrix<T>::device_pointer_type_cl out_mem_p = res.updateDevice_CL(res.GetArrayRep() + i * numRowsPerSlice * out_cols, outRows, out_cols, device, false);
				
				size_t numBlocks[2], numThreads[2];
				numThreads[0] = std::min<size_t>(out_cols, 16);
				numThreads[1] = std::min(out_rows, maxThreads / 16);
				numBlocks[0] = (size_t)((out_cols + numThreads[0] - 1) / numThreads[0]) * numThreads[0];
				numBlocks[1] = (size_t)((outRows + numThreads[1] - 1) / numThreads[1]) * numThreads[1];
				
				const size_t sharedRows = numThreads[1] + this->m_overlap_y * 2;
				const size_t sharedCols = numThreads[0] + this->m_overlap_x * 2;
				const size_t sharedMemSize =  sharedRows * sharedCols * sizeof(T);
				const size_t stride = numThreads[0] + this->m_overlap_x * 2;
			
				auto anyMemP = std::make_tuple(get<AI, CallArgs...>(args...).getParent().updateDevice_CL(get<AI, CallArgs...>(args...).getAddress(),
					get<AI, CallArgs...>(args...).getParent().size(), device, hasReadAccess(MapOverlapFunc::anyAccessMode[AI]))...);
				
				CLKernel::mapOverlap2D(
					i, numThreads, numBlocks,
					in_mem_p,
					std::make_tuple(&get<AI, CallArgs...>(args...).getParent(), std::get<AI>(anyMemP))...,
					get<CI, CallArgs...>(args...)...,
					out_mem_p, outRows, out_cols,
					this->m_overlap_y, this->m_overlap_x,
					in_cols, sharedRows, sharedCols,
					sharedMemSize
				);
				
				// Make sure the data is marked as changed by the device
				pack_expand((std::get<AI>(anyMemP)->changeDeviceData(hasWriteAccess(MapOverlapFunc::anyAccessMode[AI])), 0)...);
				out_mem_p->changeDeviceData();
			}
		}
		
		
		template<typename MapOverlapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs>
		void MapOverlap2D<MapOverlapFunc, CUDAKernel, CLKernel>
		::helper_OpenCL(skepu2::Matrix<Ret>& res, skepu2::Matrix<T>& arg, pack_indices<AI...> ai, pack_indices<CI...> ci,  CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenCL MapOverlap 2D: size = " << arg.size() << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			if (numDevices <= 1)
				return this->mapOverlapSingleThread_CL(0, res, arg, ai, ci, get<AI, CallArgs...>(args...).begin()..., get<CI, CallArgs...>(args...)...);
			else
			
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
				
				return this->mapOverlapMultipleThread_CL(numDevices, res, arg, ai, ci, get<AI, CallArgs...>(args...).begin()..., get<CI, CallArgs...>(args...)...);
		}
	}
}

#endif
