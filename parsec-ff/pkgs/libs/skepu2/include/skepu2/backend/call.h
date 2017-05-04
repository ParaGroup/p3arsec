/*! \file call.h
 *  \brief Contains a class declaration for the Call skeleton.
 */

#ifndef CALL_H
#define CALL_H

namespace skepu2
{
	namespace backend
	{
		/*!
		 *  \ingroup skeletons
 		 */
		/*!
		 *  \class Call
		 *
		 *  \brief A class representing the Call skeleton.
		 *
		 *  This class defines the Map skeleton, a calculation pattern where a user function is applied to each element of an input
		 *  range. Once the Call object is instantiated, it is meant to be used as a function and therefore overloading
		 *  \p operator(). There are several overloaded versions of this operator that can be used depending on how many elements
		 *  the mapping function uses (one, two or three). There are also variants which takes iterators as inputs and those that
		 *  takes whole containers (vectors, matrices). The container variants are merely wrappers for the functions which takes iterators as parameters.
		 */
		template<typename CallFunc, typename CUDAKernel, typename CLKernel>
		class Call : public SkeletonBase
		{
			static constexpr size_t numArgs = CallFunc::totalArity;
			static constexpr size_t anyArity = std::tuple_size<typename CallFunc::ContainerArgs>::value;
			static constexpr typename make_pack_indices<anyArity, 0>::type any_indices{};
			static constexpr typename make_pack_indices<numArgs, anyArity>::type const_indices{};
			
			CUDAKernel m_cuda_kernel;
			
		public:
			
			static constexpr auto skeletonType = SkeletonType::Call;
			using ResultArg = std::tuple<>;
			using ElwiseArgs = std::tuple<>;
			using ContainerArgs = typename CallFunc::ContainerArgs;
			using UniformArgs = typename CallFunc::UniformArgs;
			static constexpr bool prefers_matrix = false;
			
			Call(CUDAKernel kernel) : m_cuda_kernel(kernel)
			{
#ifdef SKEPU_OPENCL
				CLKernel::initialize();
#endif
			}
			
			template<typename... Args>
			void tune(Args&&... args)
			{
				tuner::tune(*this, std::forward<Args>(args)...);
			}
			
			template<typename... CallArgs>
			void operator()(CallArgs&&... args)
			{
				return backendDispatch(any_indices, const_indices, std::forward<CallArgs>(args)...);
			}
			
		private:
			
			// ==========================  CPU implementation   ==========================
			
			template<size_t... AI, size_t... CI, typename... CallArgs> 
			void  CPU(pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			
			// ========================== OpenMP implementation ==========================
#ifdef SKEPU_OPENMP
			
			template<size_t... AI, size_t... CI, typename ...CallArgs> 
			void  OMP(pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
#endif // SKEPU_OPENMP
			
			
			// ==========================  CUDA implementation  ==========================
#ifdef SKEPU_CUDA
			
			template<size_t... AI, size_t... CI, typename... CallArgs> 
			void CUDA(pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... AI, size_t... CI, typename... CallArgs> 
			void         callSingleThread_CU(size_t deviceID,  pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... AI, size_t... CI, typename... CallArgs> 
			void          callMultiStream_CU(size_t deviceID,  pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... AI, size_t... CI, typename... CallArgs> 
			void callSingleThreadMultiGPU_CU(size_t useNumGPU, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
			template<size_t... AI, size_t... CI, typename... CallArgs> 
			void  callMultiStreamMultiGPU_CU(size_t useNumGPU, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
#endif // SKEPU_CUDA
			
			
			// ========================== OpenCL implementation ==========================
#ifdef SKEPU_OPENCL
			
			template<size_t... AI, size_t... CI, typename... CallArgs> 
			void   CL(pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args);
			
#endif // SKEPU_OPENCL
			
			template<size_t... AI, size_t... CI, typename... CallArgs> 
			void backendDispatch(pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
			{
				assert(this->m_execPlan != NULL && this->m_execPlan->isCalibrated());
				
				this->m_selected_spec = (this->m_user_spec != nullptr)
					? this->m_user_spec
					: &this->m_execPlan->find(0);
				
				switch (this->m_selected_spec->backend())
				{
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					this->CUDA(ai, ci, get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
					break;
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					this->CL(ai, ci, get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
					break;
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					this->OMP(ai, ci, get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
					break;
#endif
				default:
					this->CPU(ai, ci, get<AI, CallArgs...>(args...)..., get<CI, CallArgs...>(args...)...);
					break;
				}
			}
		
		}; // class Map
		
	} // namespace backend
} // namespace skepu2


#include "impl/call/call_cpu.inl"
#include "impl/call/call_omp.inl"
#include "impl/call/call_cl.inl"
#include "impl/call/call_cu.inl"

#endif // CALL_H
