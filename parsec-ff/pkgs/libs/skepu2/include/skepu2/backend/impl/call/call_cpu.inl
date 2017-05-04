/*! \file call_cpu.inl
 *  \brief Contains the definitions of CPU specific member functions for the Call skeleton.
 */

namespace skepu2
{
	namespace backend
	{
		template<typename CallFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs> 
		void Call<CallFunc, CUDAKernel, CLKernel> 
		::CPU(pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("CPU Call");
			
			// Sync with device data
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(CallFunc::anyAccessMode[AI])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(CallFunc::anyAccessMode[AI])), 0)...);
			
			return CallFunc::CPU(get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
		}
	}
}

