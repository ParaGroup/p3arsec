/*! \file reduce.h
 *  \brief Contains a class declaration for the Reduce skeleton.
 */

#ifndef REDUCE_H
#define REDUCE_H

#include "reduce_helpers.h"

namespace skepu2
{
	enum class ReduceMode
	{
		RowWise, ColWise
	};
	
	inline std::ostream &operator<<(std::ostream &o, ReduceMode m)
	{
		switch (m)
		{
		case ReduceMode::RowWise:
			o << "Rowwise"; break;
		case ReduceMode::ColWise:
			o << "Colwise"; break;
		default:
			o << "<Invalid reduce mode>";
		}
		return o;
	}
	
	namespace backend
	{
		
		/*!
		 *  \ingroup skeletons
		 */
		/*!
		 *
		 * \brief A specilalization of above class, used for 1D Reduce operation.
		 * Please note that the class name is same. The only difference is
		 * how you instantiate it either by passing 1 user function (i.e. 1D reduction)
		 * or 2 user function (i.e. 2D reduction). See code examples for more information.
		 */
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		class Reduce1D : public SkeletonBase
		{
			using T = typename ReduceFunc::Ret;
			
		public:
			
			static constexpr auto skeletonType = SkeletonType::Reduce1D;
			using ResultArg = std::tuple<>;
			using ElwiseArgs = std::tuple<T>;
			using ContainerArgs = std::tuple<>;
			using UniformArgs = std::tuple<>;
			static constexpr bool prefers_matrix = false;
			
		public:
			Reduce1D(CUDAKernel kernel) : m_cuda_kernel(kernel)
			{
#ifdef SKEPU_OPENCL
				CLKernel::initialize();
#endif
			}
			
			void setReduceMode(ReduceMode mode)
			{
				this->m_mode = mode;
			}
			
			void setStartValue(T val)
			{
				this->m_start = val;
			}
			
			template<typename... Args>
			void tune(Args&&... args)
			{
				tuner::tune(*this, std::forward<Args>(args)...);
			}
			
		protected:
			CUDAKernel m_cuda_kernel;
			
			ReduceMode m_mode = ReduceMode::RowWise;
			T m_start{};
			
			
			Vector<T> &CPU(Vector<T> &res, Matrix<T>& arg);
			
			template<typename Iterator>
			T CPU(size_t size, T &res, Iterator arg);
			
#ifdef SKEPU_OPENMP
			
			Vector<T> &OMP(Vector<T> &res, Matrix<T>& arg);
			
			template<typename Iterator>
			T OMP(size_t size, T &res, Iterator arg);
			
#endif
			
#ifdef SKEPU_CUDA
			
			Vector<T> &reduceSingleThreadOneDim_CU(size_t deviceID, Vector<T> &res, Matrix<T> &arg);
			
			Vector<T> &reduceMultipleOneDim_CU(size_t numDevices, Vector<T> &res, Matrix<T> &arg);
			
			Vector<T> &CU(Vector<T> &res, Matrix<T>& arg);
		
			template<typename Iterator>
			T reduceSingleThread_CU(size_t deviceID, size_t size, T &res, Iterator arg);
			
			template<typename Iterator>
			T reduceMultiple_CU(size_t numDevices, size_t size, T &res, Iterator arg);
		
			template<typename Iterator>
			T CU(size_t size, T &res, Iterator arg);
#endif
			
#ifdef SKEPU_OPENCL
			
			Vector<T> &reduceSingleThreadOneDim_CL(size_t deviceID, Vector<T> &res, Matrix<T> &arg);
			
			Vector<T> &reduceMultipleOneDim_CL(size_t numDevices, Vector<T> &res, Matrix<T> &arg);
			
			Vector<T> &CL(Vector<T> &res, Matrix<T>& arg);
		
			template<typename Iterator>
			T reduceSingle_CL(size_t deviceID, size_t size, T &res, Iterator arg);
			
			template<typename Iterator>
			T reduceMultiple_CL(size_t numDevices, size_t size, T &res, Iterator arg);
		
			template<typename Iterator>
			T CL(size_t size, T &res, Iterator arg);
			
#endif
			
			
		public:
			template<template<class> class Container>
			T operator()(Container<T> &arg)
			{
				return this->backendDispatch(arg.size(), arg.begin());
			}
		
			template<typename Iterator>
			T operator()(Iterator arg, Iterator arg_end)
			{
				return this->backendDispatch(arg_end - arg, arg.begin());
			}
			
			Vector<T> &operator()(Vector<T> &res, Matrix<T>& arg)
			{
				assert(this->m_execPlan != NULL && this->m_execPlan->isCalibrated());
				
				const size_t size = arg.size();
				
				// TODO: check size
				
				this->m_selected_spec = (this->m_user_spec != nullptr)
					? this->m_user_spec
					: &this->m_execPlan->find(size);
				
				Matrix<T> &arg_tr = (this->m_mode == ReduceMode::ColWise) ? arg.transpose(*this->m_selected_spec) : arg;
				
				switch (this->m_selected_spec->backend())
				{
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					return this->CU(res, arg_tr);
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					return this->CL(res, arg_tr);
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					return this->OMP(res, arg_tr);
#endif
				default:
					return this->CPU(res, arg_tr);
				}
			}
			
		private:
			template<typename Iterator>
			T backendDispatch(size_t size, Iterator arg)
			{
				assert(this->m_execPlan != NULL && this->m_execPlan->isCalibrated());
				
				T res = this->m_start;
				
				this->m_selected_spec = (this->m_user_spec != nullptr)
					? this->m_user_spec
					: &this->m_execPlan->find(size);
				
				switch (this->m_selected_spec->backend())
				{
			case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					return this->CU(size, res, arg);
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					return this->CL(size, res, arg);
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					return this->OMP(size, res, arg);
#endif
				default:
					return this->CPU(size, res, arg);
				}
			}
			
			
		};
		
		
		/*!
		 *  \class Reduce
		 *
		 *  \brief A class representing the Reduce skeleton both for 1D and 2D reduce operation for 1D Vector, 2D Dense Matrix/Sparse matrices.
		 *
		 *  This class defines the Reduce skeleton which support following reduction operations:
		 *  (a) (1D Reduction) Each element in the input range, yielding a scalar result by applying a commutative associative binary operator.
		 *     Here we consider dense/sparse matrix as vector thus reducing all (non-zero) elements of the matrix.
		 *  (b) (1D Reduction) Dense/Sparse matrix types: Where we reduce either row-wise or column-wise by applying a commutative associative binary operator. It returns
		 *     a \em SkePU vector of results that corresponds to reduction on either dimension.
		 *  (c) (2D Reduction) Dense/Sparse matrix types: Where we reduce both row- and column-wise by applying two commutative associative
		 *     binary operators, one for row-wise reduction and one for column-wise reduction. It returns a scalar result.
		 *  Two versions of this class are created using C++ partial class-template specialization to support
		 *  (a) 1D reduction (where a "single" reduction operator is applied on all elements or to 1 direction for 2D Dense/Sparse matrix).
		 *  (b) 2D reduction that works only for matrix (where two different reduction operations are used to reduce row-wise and column-wise separately.)
		 *  Once instantiated, it is meant to be used as a function and therefore overloading
		 *  \p operator(). The Reduce skeleton needs to be created with
		 *  a 1 or 2 binary user function for 1D reduction and 2D reduction respectively.
		 */
		template<typename ReduceFuncRowWise, typename ReduceFuncColWise, typename CUDARowWise, typename CUDAColWise, typename CLKernel>
		class Reduce2D : public Reduce1D<ReduceFuncRowWise, CUDARowWise, CLKernel>
		{
			using T = typename ReduceFuncRowWise::Ret;
			
		public:
			
			static constexpr auto skeletonType = SkeletonType::Reduce2D;
			static constexpr bool prefers_matrix = true;
			
			Reduce2D(CUDARowWise row, CUDAColWise col) : Reduce1D<ReduceFuncRowWise, CUDARowWise, CLKernel>(row), m_cuda_colwise_kernel(col) {}
			
		private:
			CUDAColWise m_cuda_colwise_kernel;
			
			
			
			
		private:
			T CPU(T &res, Matrix<T>& arg);
			
#ifdef SKEPU_OPENMP
			
			T OMP(T &res, Matrix<T>& arg);
			
			T ompVectorReduce(T &res, std::vector<T> &input, size_t numThreads);
			
#endif
			
#ifdef SKEPU_CUDA
			
			T CU(T &res, Matrix<T>& arg);
			
			T reduceSingleThread_CU(size_t deviceID, T &res, Matrix<T>& arg);
			
			T reduceMultiple_CU(size_t numDevices, T &res, Matrix<T>& arg);
			
#endif
			
#ifdef SKEPU_OPENCL
			
			T CL(T &res, Matrix<T>& arg);
			
			T reduceSingle_CL(size_t deviceID, T &res, Matrix<T>& arg);
			
			T reduceNumDevices_CL(size_t numDevices, T &res, Matrix<T>& arg);
			
#endif
			
			
			
		public:
			T operator()(Vector<T>& arg)
			{
				return Reduce1D<ReduceFuncRowWise, CUDARowWise, CLKernel>::operator()(arg);
			}
			
			T operator()(Matrix<T>& arg)
			{
				assert(this->m_execPlan != NULL && this->m_execPlan->isCalibrated());
				
				this->m_selected_spec = (this->m_user_spec != nullptr)
					? this->m_user_spec
					: &this->m_execPlan->find(arg.size());
					
				T res = this->m_start;
				
				Matrix<T> &arg_tr = (this->m_mode == ReduceMode::ColWise) ? arg.transpose(*this->m_selected_spec) : arg;
				
				switch (this->m_selected_spec->backend())
				{
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					return this->CU(res, arg_tr);
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					return this->CL(res, arg_tr);
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					return this->OMP(res, arg_tr);
#endif
				default:
					return this->CPU(res, arg_tr);
				}
			}
			
		};
		
	} // end namespace backend
} // end namespace skepu2


#include "impl/reduce/reduce_cpu.inl"
#include "impl/reduce/reduce_omp.inl"
#include "impl/reduce/reduce_cl.inl"
#include "impl/reduce/reduce_cu.inl"

#endif // REDUCE_H
