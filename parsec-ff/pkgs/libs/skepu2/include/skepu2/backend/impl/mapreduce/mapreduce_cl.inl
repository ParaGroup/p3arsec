/*! \file mapreduce_cl.inl
 *  \brief Contains the definitions of OpenCL specific member functions for the MapReduce skeleton.
 */

#ifdef SKEPU_OPENCL

namespace skepu2
{
	namespace backend
	{
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs> 
		typename ReduceFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::mapReduceSingle_CL(size_t deviceID, size_t size, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, Ret &res, CallArgs&&... args)
		{
			auto eArgs  = std::make_tuple(get<EI, CallArgs...>(args...)...);
			auto aArgs  = std::make_tuple(get<AI, CallArgs...>(args...)...);
			
			const size_t numThreads = this->m_selected_spec->GPUThreads();
			const size_t numBlocks = std::max<size_t>(1, std::min(size / numThreads, this->m_selected_spec->GPUBlocks()));
			Device_CL *device = this->m_environment->m_devices_CL[deviceID];
			
			DEBUG_TEXT_LEVEL1("OpenCL MapReduce: numThreads = " << numThreads << ", numBlocks = " << numBlocks);
			
			// Copies the elements to the device
			auto elwiseMemP = std::make_tuple(std::get<EI>(eArgs).getParent().updateDevice_CL(get<EI, CallArgs...>(args...).getAddress(), size, device, true)...);
			auto anyMemP    = std::make_tuple(std::get<AI-arity>(aArgs).getParent().updateDevice_CL(get<AI, CallArgs...>(args...).getAddress(),
				get<AI, CallArgs...>(args...).size(), device, hasReadAccess(MapFunc::anyAccessMode[AI-arity]))...);
			
			// Create the output memory
			DeviceMemPointer_CL<Ret> outMemP(&res, numBlocks, device);
			
			CLKernel::mapReduce (
				deviceID, numThreads, numBlocks * numThreads,
				device_mem_pointer_const_cast(std::get<EI>(elwiseMemP))...,
				std::make_tuple(&get<AI, CallArgs...>(args...).getParent(), std::get<AI-arity>(anyMemP))...,
				get<CI, CallArgs...>(args...)...,
				&outMemP,
				elwise_width(eArgs),
				size,
				0,
				sizeof(Ret) * numThreads
			);
			
			CLKernel::reduceOnly (
				deviceID, numThreads, numThreads,
				&outMemP, &outMemP, numBlocks, sizeof(Ret) * numThreads
			);
			
			// Copy back result
			pack_expand((std::get<AI-arity>(anyMemP)->changeDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			outMemP.changeDeviceData();
			outMemP.copyDeviceToHost(1);
			
			return res;
		}
		
		
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs> 
		typename ReduceFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::mapReduceNumDevices_CL(size_t numDevices, size_t size, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, Ret &res, CallArgs&&... args)
		{
			auto eArgs  = std::make_tuple(get<EI, CallArgs...>(args...)...);
			auto aArgs  = std::make_tuple(get<AI, CallArgs...>(args...)...);
			
			const size_t numElemPerSlice = size / numDevices;
			const size_t rest = size % numDevices;
			
			Ret result[numDevices];
			std::vector<DeviceMemPointer_CL<Ret>> outMemP;
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numElem = numElemPerSlice + (i == numDevices - 1 ? rest : 0);
				const size_t numThreads = this->m_selected_spec->GPUThreads();
				const size_t numBlocks = std::max<size_t>(1, std::min(numElem / numThreads, this->m_selected_spec->GPUBlocks()));
				const size_t baseIndex = i * numElemPerSlice;
				
				DEBUG_TEXT_LEVEL1("OpenCL MapReduce: device = " << i << ", numThreads = " << numThreads << ", numBlocks = " << numBlocks);
				
				// Copies the elements to the device
				auto elwiseMemP = std::make_tuple(std::get<EI>(eArgs).getParent().updateDevice_CL(get<EI, CallArgs...>(args...).getAddress() + baseIndex, numElem, device, true)...);
				auto anyMemP    = std::make_tuple(std::get<AI-arity>(aArgs).getParent().updateDevice_CL(get<AI, CallArgs...>(args...).getAddress(),
					get<AI, CallArgs...>(args...).size(), device, hasReadAccess(MapFunc::anyAccessMode[AI-arity]))...);
			
				// Create the output memory
				outMemP.emplace_back(&result[i], numBlocks, device);
				
				CLKernel::mapReduce (
					i, numThreads, numBlocks * numThreads,
					device_mem_pointer_const_cast(std::get<EI>(elwiseMemP))...,
					std::make_tuple(&get<AI, CallArgs...>(args...).getParent(), std::get<AI-arity>(anyMemP))...,
					get<CI, CallArgs...>(args...)...,
					&outMemP[i],
					elwise_width(eArgs),
					numElem,
					baseIndex,
					sizeof(Ret) * numThreads
				);
				
				CLKernel::reduceOnly (
					i, numThreads, numThreads,
					&outMemP[i], &outMemP[i], numBlocks, sizeof(Ret) * numThreads
				);
				
				// Copy back result
				pack_expand((std::get<AI-arity>(anyMemP)->changeDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
				outMemP[i].changeDeviceData();
			}
			
			// Reduces results from each device on the CPU to yield the total result.
			for (size_t i = 0; i < numDevices; ++i)
			{
				outMemP[i].copyDeviceToHost(1);
				res = ReduceFunc::CPU(res, result[i]);
			}
			
			return res;
		}
		
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename... CallArgs> 
		typename ReduceFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::CL(size_t size, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, Ret &res, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenCL MapReduce: size = " << size << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			if (numDevices <= 1)
				return mapReduceSingle_CL(0, size, elwise_indices, ai, ci, res, args...);
			
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
				
			return mapReduceNumDevices_CL(numDevices, size, elwise_indices, ai, ci, res, args...);
		}
		
	} // end namespace backend
} // end namespace skepu2

#endif
