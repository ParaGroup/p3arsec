/*! \file map_cpu.inl
 *  \brief Contains the definitions of CPU specific member functions for the Map skeleton.
 */

namespace skepu2
{
	namespace backend
	{
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename Iterator, typename... CallArgs> 
		void Map<arity, MapFunc, CUDAKernel, CLKernel> 
		::CPU(size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("CPU Map: size = " << size);
			
			// Sync with device data
			pack_expand((get<EI, CallArgs...>(args...).getParent().updateHost(), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			res.getParent().invalidateDeviceData();
			
			for (size_t i = 0; i < size; ++i)
			{
				auto index = (res + i).getIndex();
				res(i) = F::forward(MapFunc::CPU, index, get<EI, CallArgs...>(args...)(i)..., get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
			}
		}
	}
}

