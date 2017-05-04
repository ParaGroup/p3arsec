/*! \file map_omp.inl
 *  \brief Contains the definitions of OpenMP specific member functions for the Map skeleton.
 */

#ifdef SKEPU_OPENMP

#include <omp.h>

namespace skepu2
{
	namespace backend
	{
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename Iterator, typename... CallArgs> 
		void Map<arity, MapFunc, CUDAKernel, CLKernel>
		::OMP(size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("OpenMP Map: size = " << size);
			
			// Sync with device data
			pack_expand((get<EI, CallArgs...>(args...).getParent().updateHost(), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			res.getParent().invalidateDeviceData();
			
			omp_set_num_threads(this->m_selected_spec->CPUThreads());
			
#pragma omp parallel for
			for (size_t i = 0; i < size; ++i)
			{
				res(i) = F::forward(MapFunc::OMP, (res + i).getIndex(), get<EI, CallArgs...>(args...)(i)..., get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
			}
		}
	}
}

#endif // SKEPU_OPENMP