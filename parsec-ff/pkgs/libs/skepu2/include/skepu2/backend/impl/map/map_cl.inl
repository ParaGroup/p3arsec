/*! \file map_cl.inl
 *  \brief Contains the definitions of OpenCL specific member functions for the Map skeleton.
 */

#ifdef SKEPU_OPENCL

namespace skepu2
{
	namespace backend
	{
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename Iterator, typename... CallArgs> 
		void Map<arity, MapFunc, CUDAKernel, CLKernel>
		::mapNumDevices_CL(size_t numDevices, size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args)
		{
			constexpr size_t numKernelArgs = numArgs + 3;
			const size_t numElemPerSlice = size / numDevices;
			const size_t rest = size % numDevices;
			
			std::vector<DeviceMemPointer_CL<T>*> outMemP(numDevices);
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numElem = numElemPerSlice + ((i == numDevices-1) ? rest : 0);
				
				pack_expand((get<EI, CallArgs...>(args...).getParent().updateDevice_CL(get<EI, CallArgs...>(args...).getAddress() + i * numElemPerSlice, numElem,        device, false), 0)...);
				pack_expand((get<AI, CallArgs...>(args...).getParent().updateDevice_CL(get<AI, CallArgs...>(args...).getAddress(), get<AI, CallArgs...>(args...).size(), device, false), 0)...);
				outMemP[i] = res.getParent().updateDevice_CL(res.getAddress() + i * numElemPerSlice, numElem, device, false);
			}
			
			DEBUG_TEXT_LEVEL1("OpenCL Map: numDevices " << numDevices);
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				Device_CL *device = this->m_environment->m_devices_CL[i];
				const size_t numElem = numElemPerSlice + ((i == numDevices-1) ? rest : 0);
				const size_t numThreads = std::min(this->m_selected_spec->GPUThreads(), numElem);
				const size_t numBlocks = std::min(numElem / numThreads + (numElem % numThreads == 0 ? 0 : 1), this->m_selected_spec->GPUBlocks());
				
				DEBUG_TEXT_LEVEL1("OpenCL Map: device " << i << ", numElem = " << numElem << ", numBlocks = " << numBlocks << ", numThreads = " << numThreads);
				
				// Copies the elements to the device
				auto elwiseMemP = std::make_tuple(get<EI, CallArgs...>(args...).getParent().updateDevice_CL(get<EI, CallArgs...>(args...).getAddress() + i * numElemPerSlice, numElem, device, true)...);
				auto anyMemP = std::make_tuple(cl_helpers::randomAccessArg(get<AI, CallArgs...>(args...).getParent(), device, hasReadAccess(MapFunc::anyAccessMode[AI-arity]))...);
				
				CLKernel::map(
					i, numThreads, numBlocks * numThreads,
					std::get<EI>(elwiseMemP)...,
					std::get<AI-arity>(anyMemP)...,
					get<CI, CallArgs...>(args...)...,
					outMemP[i],
					res.getParent().total_cols(),
					numElem,
					i * numElemPerSlice
				);
				
				// Make sure the data is marked as changed by the device
				pack_expand((std::get<1>(std::get<AI-arity>(anyMemP))->changeDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
				outMemP[i]->changeDeviceData();
			}
		}
		
		
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename Iterator, typename... CallArgs> 
		void Map<arity, MapFunc, CUDAKernel, CLKernel>
		::CL(size_t size, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, Iterator res, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenCL Map: size = " << size << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
			// Same method is invoked no matter how many GPUs we use
			this->mapNumDevices_CL(numDevices, size, ei, ai, ci, res, args...);
		}
		
	} // namespace backend
} // namespace skepu2

#endif
