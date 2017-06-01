/*! \file call_cl.inl
 *  \brief Contains the definitions of OpenCL specific member functions for the Call skeleton.
 */

#ifdef SKEPU_OPENCL

namespace skepu2
{
	namespace backend
	{
		template<typename CallFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs> 
		void Call<CallFunc, CUDAKernel, CLKernel>
		::CL(pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenCL Call: Devices = " << this->m_selected_spec->devices()
				<< ", blocks = " << this->m_selected_spec->GPUBlocks() << ", threads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CL.size());
			
			for (size_t i = 0; i < numDevices; ++i)
				pack_expand((get<AI, CallArgs...>(args...).getParent().updateDevice_CL(get<AI, CallArgs...>(args...).getAddress(),
					get<AI, CallArgs...>(args...).size(), this->m_environment->m_devices_CL[i], false), 0)...);
			
			for (size_t i = 0; i < numDevices; ++i)
			{
				const size_t numThreads = this->m_selected_spec->GPUThreads();
				const size_t numBlocks = this->m_selected_spec->GPUBlocks();
				
				DEBUG_TEXT_LEVEL1("OpenCL Call: device " << i << ", numBlocks = " << numBlocks << ", numThreads = " << numThreads);
				
				// Copies the elements to the device
				auto anyMemP = std::make_tuple(get<AI, CallArgs...>(args...).getParent().updateDevice_CL(get<AI, CallArgs...>(args...).getAddress(),
					get<AI, CallArgs...>(args...).size(), this->m_environment->m_devices_CL[i], hasReadAccess(CallFunc::anyAccessMode[AI]))...);
				
				CLKernel::call(
					i, numThreads, numBlocks * numThreads, 
					std::make_tuple(&get<AI, CallArgs...>(args...).getParent(), std::get<AI>(anyMemP))...,
					get<CI, CallArgs...>(args...)...
				);
				
				// Make sure the data is marked as changed by the device
				pack_expand((std::get<AI>(anyMemP)->changeDeviceData(hasWriteAccess(CallFunc::anyAccessMode[AI])), 0)...);
			}
		}
		
	} // namespace backend
} // namespace skepu2

#endif
