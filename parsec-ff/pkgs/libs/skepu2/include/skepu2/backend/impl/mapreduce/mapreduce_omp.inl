/*! \file mapreduce_omp.inl
*  \brief Contains the definitions of OpenMP specific member functions for the MapReduce skeleton.
*/

#ifdef SKEPU_OPENMP

#include <omp.h>
#include <iostream>
#include <vector>

namespace skepu2
{
	namespace backend
	{
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs> 
		typename ReduceFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::OMP(size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args)
		{
			// Sync with device data
			pack_expand((get<EI, CallArgs...>(args...).getParent().updateHost(), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			
			// Set up thread indexing
			omp_set_num_threads(std::min(this->m_selected_spec->CPUThreads(), size / 2));
			const size_t nthr = omp_get_max_threads();
			const size_t q = size / nthr;
			const size_t rest = size % nthr;
			
			std::vector<Ret> parsums(nthr);
			
			// Perform Map and partial Reduce with OpenMP
#pragma omp parallel
			{
				const size_t myid = omp_get_thread_num();
				const size_t first = myid * q;
				const size_t last = (myid + 1) * q + (myid == nthr - 1 ? rest : 0);
				
				Ret psum = F::forward(MapFunc::OMP, (get<0, CallArgs...>(args...) + first).getIndex(), get<EI, CallArgs...>(args...)(first)..., get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
				
				for (size_t i = first+1; i < last; ++i)
				{
					Temp tempMap = F::forward(MapFunc::OMP, (get<0, CallArgs...>(args...) + i).getIndex(), get<EI, CallArgs...>(args...)(i)..., get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
					psum = ReduceFunc::OMP(psum, tempMap);
				}
				parsums[myid] = psum;
			}
			
			// Final Reduce sequentially
			for (Ret &parsum : parsums)
				res = ReduceFunc::OMP(res, parsum);
			
			return res;
		}
		
		
		
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... AI, size_t... CI, typename ...CallArgs> 
		typename ReduceFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::OMP(size_t size, pack_indices<>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args)
		{
			// Sync with device data
			pack_expand((get<AI, CallArgs...>(args...).getParent().updateHost(hasReadAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			pack_expand((get<AI, CallArgs...>(args...).getParent().invalidateDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			
			// Set up thread indexing
			omp_set_num_threads(std::min(this->m_selected_spec->CPUThreads(), size / 2));
			const size_t nthr = omp_get_max_threads();
			const size_t q = size / nthr;
			const size_t rest = size % nthr;
			
			std::vector<Ret> parsums(nthr);
			
			// Perform Map and partial Reduce with OpenMP
#pragma omp parallel
			{
				const size_t myid = omp_get_thread_num();
				const size_t first = myid * q;
				const size_t last = (myid + 1) * q + (myid == nthr - 1 ? rest : 0);
				
				Ret psum = F::forward(MapFunc::OMP, skepu2::Index1D{first}, get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
				
				for (size_t i = first+1; i < last; ++i)
				{
					Temp tempMap = F::forward(MapFunc::OMP, skepu2::Index1D{i}, get<AI, CallArgs...>(args...).hostProxy()..., get<CI, CallArgs...>(args...)...);
					psum = ReduceFunc::OMP(psum, tempMap);
				}
				parsums[myid] = psum;
			}
			
			// Final Reduce sequentially
			for (Ret &parsum : parsums)
				res = ReduceFunc::OMP(res, parsum);
			
			return res;
		}
		
	} // namespace backend
} // namespace skepu2

#endif

