/*! \file mapreduce.h
 *  \brief Contains a class declaration for the MapReduce skeleton.
 */

#ifndef MAPREDUCE_H
#define MAPREDUCE_H

namespace skepu2
{
	namespace backend
	{
		/*!
		*  \ingroup skeletons
		*/
		/*!
		*  \class MapReduce
		*
		*  \brief A class representing the MapReduce skeleton.
		*
		*  This class defines the MapReduce skeleton which is a combination of the Map and Reduce operations. It produces
		*  the same result as if one would first Map one or more vectors (matrices) to a result vector (matrix), then do a reduction on that result.
		*  It is provided since it combines the mapping and reduction in the same computation kernel and therefore avoids some
		*  synchronization, which speeds up the calculation. Once instantiated, it is meant to be used as a function and therefore overloading
		*  \p operator(). There are several overloaded versions of this operator that can be used depending on how many elements
		*  the mapping function uses (one, two or three). There are also variants which takes iterators as inputs and those that
		*  takes whole containers (vectors, matrices).
		*/
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		class MapReduce : public SkeletonBase
		{
		public:
			MapReduce(CUDAKernel mapreduce, CUDAReduceKernel reduce)
			: m_cuda_kernel(mapreduce), m_cuda_reduce_kernel(reduce)
			{
#ifdef SKEPU_OPENCL
				CLKernel::initialize();
#endif
			}
			
			static constexpr auto skeletonType = SkeletonType::MapReduce;
			using ResultArg = std::tuple<>;
			using ElwiseArgs = typename MapFunc::ElwiseArgs;
			using ContainerArgs = typename MapFunc::ContainerArgs;
			using UniformArgs = typename MapFunc::UniformArgs;
			static constexpr bool prefers_matrix = MapFunc::prefersMatrix;
			
		private:
			CUDAKernel m_cuda_kernel;
			CUDAReduceKernel m_cuda_reduce_kernel;
			size_t default_size_x;
			size_t default_size_y;
			
			static constexpr size_t numArgs = MapFunc::totalArity - (MapFunc::indexed ? 1 : 0);
			static constexpr size_t anyArity = std::tuple_size<typename MapFunc::ContainerArgs>::value;
			static constexpr typename make_pack_indices<arity, 0>::type elwise_indices{};
			static constexpr typename make_pack_indices<arity + anyArity, arity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, arity + anyArity>::type const_indices{};
			
			using First = typename parameter_type<MapFunc::indexed ? 1 : 0, decltype(&MapFunc::CPU)>::type;
			
			using F = ConditionalIndexForwarder<MapFunc::indexed, decltype(&MapFunc::CPU)>;
			using Temp = typename MapFunc::Ret;
			using Ret = typename ReduceFunc::Ret;
			
			Ret m_start{};
			
#pragma mark - Backend agnostic
			
		public:
			void setStartValue(Ret val)
			{
				this->m_start = val;
			}
			
			void setDefaultSize(size_t x, size_t y = 0)
			{
				this->default_size_x = x;
				this->default_size_y = y;
			}
			
			template<typename... Args>
			void tune(Args&&... args)
			{
				tuner::tune(*this, std::forward<Args>(args)...);
			}
			
			template<template<class> class Container, typename... CallArgs, REQUIRES(is_skepu_container<Container<First>>::value)>
			Ret operator()(const Container<First> &arg1, CallArgs&&... args)
			{
				return backendDispatch(elwise_indices, any_indices, const_indices, arg1.size(), arg1, std::forward<CallArgs>(args)...);
			}
			
			template<template<class> class Container, typename... CallArgs, REQUIRES(is_skepu_container<Container<First>>::value)>
			Ret operator()(Container<First> &arg1, CallArgs&&... args)
			{
				return backendDispatch(elwise_indices, any_indices, const_indices, arg1.size(), arg1, std::forward<CallArgs>(args)...);
			}
			
			template<typename Iterator, typename... CallArgs, REQUIRES(is_skepu_iterator<Iterator, First>::value)>
			Ret operator()(Iterator arg1, Iterator arg1_end, CallArgs&&... args)
			{
				return backendDispatch(elwise_indices, any_indices, const_indices, arg1_end - arg1, arg1, std::forward<CallArgs>(args)...);
			}
			
			template<template<class> class Container = Vector, typename... CallArgs>
			Ret operator()(CallArgs&&... args)
			{
				static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching Map function");
				
			//	if (this->default_size_y != 0)
					return this->backendDispatch(elwise_indices, any_indices, const_indices, this->default_size_x, std::forward<CallArgs>(args)...);
				
			//	else
			//		return this->zero_apply_2D(any_indices, const_indices, this->default_size_x, this->default_size_y, std::forward<CallArgs>(args)...);
			}
			
		private:
			template<size_t... EI, size_t... AI, size_t... CI, typename... CallArgs> 
			Ret backendDispatch(pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, size_t size, CallArgs&&... args)
			{
				assert(this->m_execPlan != NULL && this->m_execPlan->isCalibrated());
				
				Ret res = this->m_start;
				
				if (disjunction((get<EI, CallArgs...>(args...).size() < size)...))
					SKEPU_ERROR("Non-matching container sizes");
				
				this->m_selected_spec = (this->m_user_spec != nullptr)
					? this->m_user_spec
					: &this->m_execPlan->find(size);
				
				switch (this->m_selected_spec->backend())
				{
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					return CUDA(size, ei, ai, ci, res, get<EI, CallArgs...>(args...).begin()..., get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					return   CL(size, ei, ai, ci, res, get<EI, CallArgs...>(args...).begin()..., get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					return  OMP(size, ei, ai, ci, res, get<EI, CallArgs...>(args...).begin()..., get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
#endif
				default:
					return  CPU(size, ei, ai, ci, res, get<EI, CallArgs...>(args...).begin()..., get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
				}
			}
			
			
			template<size_t... EI, size_t... AI, size_t... CI, typename... CallArgs> 
			Ret CPU(size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
			template<size_t... AI, size_t... CI, typename... CallArgs> 
			Ret CPU(size_t size, pack_indices<>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
			
#ifdef SKEPU_OPENMP
			
			template<size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs> 
			Ret OMP(size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
			template<size_t... AI, size_t... CI, typename ...CallArgs> 
			Ret OMP(size_t size, pack_indices<>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
#endif // SKEPU_OPENMP
			
#ifdef SKEPU_CUDA
			
			template<size_t... EI, size_t... AI, size_t... CI, typename... CallArgs> 
			Ret CUDA(size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
			template<size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs> 
			Ret mapReduceSingleThread_CU(size_t deviceID, size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
			template<size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs> 
			Ret mapReduceMultiStream_CU(size_t deviceID, size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
			template<size_t... EI, size_t... AI, size_t... CI, typename... CallArgs> 
			Ret mapReduceSingleThreadMultiGPU_CU(size_t useNumGPU, size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
			template<size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs> 
			Ret mapReduceMultiStreamMultiGPU_CU(size_t useNumGPU, size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
			
#endif // SKEPU_CUDA
			
#ifdef SKEPU_OPENCL
			
			template<size_t... EI, size_t... AI, size_t... CI, typename... CallArgs> 
			Ret CL(size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
			template<size_t... EI, size_t... AI, size_t... CI, typename... CallArgs> 
			Ret mapReduceNumDevices_CL(size_t numDevices, size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
			template<size_t... EI, size_t... AI, size_t... CI, typename... CallArgs> 
			Ret mapReduceSingle_CL(size_t deviceID, size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args);
			
#endif // SKEPU_OPENCL
		
		}; // class MapReduce
		
	} // end namespace backend
} // end namespace skepu2


#include "impl/mapreduce/mapreduce_cpu.inl"
#include "impl/mapreduce/mapreduce_omp.inl"
#include "impl/mapreduce/mapreduce_cl.inl"
#include "impl/mapreduce/mapreduce_cu.inl"

#endif
