/*! \file map.h
 *  \brief Contains a class declaration for the Map skeleton.
 */

#ifndef MAP_H
#define MAP_H

namespace skepu2
{
	namespace backend
	{
		/*!
		 *  \ingroup skeletons
 		 */
		/*!
		 *  \class Map
		 *
		 *  \brief A class representing the Map skeleton.
		 *
		 *  This class defines the Map skeleton, a calculation pattern where a user function is applied to each element of an input
		 *  range. Once the Map object is instantiated, it is meant to be used as a function and therefore overloading
		 *  \p operator(). There are several overloaded versions of this operator that can be used depending on how many elements
		 *  the mapping function uses (one, two or three). There are also variants which takes iterators as inputs and those that
		 *  takes whole containers (vectors, matrices). The container variants are merely wrappers for the functions which takes iterators as parameters.
		 */
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		class Map : public SkeletonBase
		{
			// ==========================    Type definitions   ==========================
			
			using T = typename MapFunc::Ret;
			using F = ConditionalIndexForwarder<MapFunc::indexed, decltype(&MapFunc::CPU)>;
			
			// ==========================     Class members     ==========================
			
			static constexpr size_t numArgs = MapFunc::totalArity - (MapFunc::indexed ? 1 : 0);
			static constexpr size_t anyArity = std::tuple_size<typename MapFunc::ContainerArgs>::value;
			
			// ==========================    Instance members   ==========================
			
			CUDAKernel m_cuda_kernel;
			
			size_t default_size_x;
			size_t default_size_y;
			
		public:
			
			static constexpr auto skeletonType = SkeletonType::Map;
			using ResultArg = std::tuple<T>;
			using ElwiseArgs = typename MapFunc::ElwiseArgs;
			using ContainerArgs = typename MapFunc::ContainerArgs;
			using UniformArgs = typename MapFunc::UniformArgs;
			static constexpr bool prefers_matrix = MapFunc::prefersMatrix;
			
			static constexpr typename make_pack_indices<arity, 0>::type elwise_indices{};
			static constexpr typename make_pack_indices<arity + anyArity, arity>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, arity + anyArity>::type const_indices{};
			
			// =========================      Constructors      ==========================
			
			Map(CUDAKernel kernel) : m_cuda_kernel(kernel)
			{
#ifdef SKEPU_OPENCL
				CLKernel::initialize();
#endif
			}
			
			// =======================  Persistent parameters   ==========================
			
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
			
			// =======================      Call operators      ==========================
			
			template<template<class> class Container, typename... CallArgs, REQUIRES(is_skepu_container<Container<T>>::value)>
			Container<T> &operator()(Container<T> &res, CallArgs&&... args)
			{
				this->backendDispatch(elwise_indices, any_indices, const_indices, res.size(), res.begin(), args...);
				return res;
			}
			
			
			template<typename Iterator, typename... CallArgs, REQUIRES(is_skepu_iterator<Iterator, T>::value)>
			Iterator operator()(Iterator res, Iterator res_end, CallArgs&&... args)
			{
				this->backendDispatch(elwise_indices, any_indices, const_indices, res_end - res, res, args...);
				return res;
			}
			
			template<template<class> class Container = Vector, typename... CallArgs>
			Container<T> operator()(CallArgs&&... args)
			{
				static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching Map function");
				
			/*	if (this->default_size_y != 0)
				{
					Container<Ret> res(this->default_size_x, this->default_size_y);
					this->apply(elwise_indices, any_indices, const_indices, res, std::forward<CallArgs>(args)...);
					return std::move(res);
				}
				else
				{*/
					Container<T> res(this->default_size_x);
					this->backendDispatch(elwise_indices, any_indices, const_indices, res.size(), res.begin(), std::forward<CallArgs>(args)...);
					return std::move(res);
			//	}
			}
			
		private:
			
			// ==========================    Implementation     ==========================
			
			template<size_t... EI, size_t... AI, size_t... CI, typename Iterator, typename... CallArgs> 
			void CPU(size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args);
			
			
#ifdef SKEPU_OPENMP
			
			template<size_t... EI, size_t... AI, size_t... CI, typename Iterator, typename ...CallArgs> 
			void OMP(size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args);
			
#endif // SKEPU_OPENMP
			
			
#ifdef SKEPU_CUDA
			
			template<size_t... EI, size_t... AI, size_t... CI, typename Iterator, typename... CallArgs> 
			void CUDA(size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args);
			
			template<size_t... EI, size_t... AI, size_t... CI, typename Iterator, typename ...CallArgs> 
			void         mapSingleThread_CU(size_t deviceID,  size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args);
			
			template<size_t... EI, size_t... AI, size_t... CI, typename Iterator, typename ...CallArgs> 
			void          mapMultiStream_CU(size_t deviceID,  size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args);
			
			template<size_t... EI, size_t... AI, size_t... CI, typename Iterator, typename... CallArgs> 
			void mapSingleThreadMultiGPU_CU(size_t useNumGPU, size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args);
			
			template<size_t... EI, size_t... AI, size_t... CI, typename Iterator, typename ...CallArgs> 
			void  mapMultiStreamMultiGPU_CU(size_t useNumGPU, size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args);
			
#endif // SKEPU_CUDA
			
			
#ifdef SKEPU_OPENCL
			
			template<size_t... EI, size_t... AI, size_t... CI, typename Iterator, typename... CallArgs> 
			void CL(size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args);
			
			template<size_t... EI, size_t... AI, size_t... CI, typename Iterator, typename... CallArgs> 
			void mapNumDevices_CL(size_t numDevices, size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args);
			
#endif // SKEPU_OPENCL
			
			
			template<size_t... EI, size_t... AI, size_t... CI, typename Iterator, typename... CallArgs> 
			void backendDispatch(pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, size_t size, Iterator res, CallArgs&&... args)
			{
				assert(this->m_execPlan != nullptr && this->m_execPlan->isCalibrated());
				
				if (disjunction((get<EI, CallArgs...>(args...).size() < size)...))
					SKEPU_ERROR("Map: Non-matching container sizes");
				
				this->m_selected_spec = (this->m_user_spec != nullptr)
					? this->m_user_spec
					: &this->m_execPlan->find(size);
				
				switch (this->m_selected_spec->backend())
				{
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					this->CUDA(size, ei, ai, ci, res, get<EI, CallArgs...>(args...).begin()..., get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
					break;
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					this->CL(size, ei, ai, ci, res, get<EI, CallArgs...>(args...).begin()..., get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
					break;
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					this->OMP(size, ei, ai, ci, res, get<EI, CallArgs...>(args...).begin()..., get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
					break;
#endif
				default:
					this->CPU(size, ei, ai, ci, res, get<EI, CallArgs...>(args...).begin()..., get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
					break;
				}
			}
		
		}; // class Map
		
	} // namespace backend
} // namespace skepu2


#include "impl/map/map_cpu.inl"
#include "impl/map/map_omp.inl"
#include "impl/map/map_cl.inl"
#include "impl/map/map_cu.inl"

#endif // MAP_H
