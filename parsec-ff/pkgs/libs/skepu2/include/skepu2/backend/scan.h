/*! \file scan.h
 *  \brief Contains a class declaration for the Scan skeleton.
 */

#ifndef SCAN_H
#define SCAN_H

namespace skepu2
{
	enum class ScanMode
	{
		Inclusive, Exclusive
	};
	
	namespace backend
	{
		/*!
		 *  \ingroup skeletons
		 */
		/*!
		 *  \class Scan
		 *
		 *  \brief A class representing the Scan skeleton.
		 *
		 *  This class defines the Scan skeleton, also known as prefix sum. It is related to the Scan operation
		 *  but instead of producing a single scalar result it produces an output vector of the same length as the
		 *  input with its elements being the reduction of itself all elements preceding it in the input. For example the
		 *  input vector [4 3 7 6 9] would produce the result vector [4 7 14 20 29]. The Scan operation can either include
		 *  or exclude the current element. It can be either inclusive or exclusive. In the previous example a inclusive
		 *  scan was performed, the exclusive result would be [0 4 7 14 20]. Exclusive scan is sometimes called prescan.
		 *  This Scan skeleton supports both variants by adding a parameter to the function calls, default is inclusive.
		 *
		 *  Once instantiated, it is meant to be used as a function and therefore overloading
		 *  \p operator(). There are a few overloaded variants of this operator depending on if a seperate output vector is provided
		 *  or if vectors or iterators are used as parameters.
		 */
		template <typename ScanFunc, typename CUDAScan, typename CUDAScanUpdate, typename CUDAScanAdd, typename CLKernel>
		class Scan : public SkeletonBase
		{
			using T = typename ScanFunc::Ret;
			
		public:
			
			static constexpr auto skeletonType = SkeletonType::Scan;
			using ResultArg = std::tuple<T>;
			using ElwiseArgs = std::tuple<T>;
			using ContainerArgs = std::tuple<>;
			using UniformArgs = std::tuple<>;
			static constexpr bool prefers_matrix = false;
			
			Scan(CUDAScan scan, CUDAScanUpdate update, CUDAScanAdd add)
			: m_cuda_scan_kernel(scan), m_cuda_scan_update_kernel(update), m_cuda_scan_add_kernel(add)
			{
#ifdef SKEPU_OPENCL
				CLKernel::initialize();
#endif
			}
			
			void setScanMode(ScanMode mode)
			{
				this->m_mode = mode;
			}
			
			void setStartValue(T initial)
			{
				this->m_initial = initial;
			}
			
			template<typename... Args>
			void tune(Args&&... args)
			{
				tuner::tune(*this, std::forward<Args>(args)...);
			}
			
		private:
			CUDAScan m_cuda_scan_kernel;
			CUDAScanUpdate m_cuda_scan_update_kernel;
			CUDAScanAdd m_cuda_scan_add_kernel;
			
			ScanMode m_mode {ScanMode::Inclusive};
			
			// Default initial value is a default-initialized value
			T m_initial {};
			
			
#pragma mark - Backend agnostic
		public:
			
			template<typename Iterator, typename In>
			Iterator operator()(Iterator res, Iterator res_end, In&& arg)
			{
				this->backendDispatch(res_end - res, res, arg.begin());
				return res;
			}
			
			template<template<class> class Container, typename In>
			Container<T>& operator()(Container<T>& res, In&& arg)
			{
				this->backendDispatch(res.size(), res.begin(), arg.begin());
				return res;
			}
			
		private:
			template<typename OutIterator, typename InIterator>
			void backendDispatch(size_t size, OutIterator res, InIterator arg)
			{
				assert(this->m_execPlan != NULL && this->m_execPlan->isCalibrated());
				
				if (arg.size() < size)
					SKEPU_ERROR("Map: Non-matching container sizes");
				
				this->m_selected_spec = (this->m_user_spec != nullptr)
					? this->m_user_spec
					: &this->m_execPlan->find(size);
				
				switch (this->m_selected_spec->backend())
				{
				case Backend::Type::CUDA:
#ifdef SKEPU_CUDA
					this->CU(size, res, arg, this->m_mode, this->m_initial);
					break;
#endif
				case Backend::Type::OpenCL:
#ifdef SKEPU_OPENCL
					this->CL(size, res, arg, this->m_mode, this->m_initial);
					break;
#endif
				case Backend::Type::OpenMP:
#ifdef SKEPU_OPENMP
					this->OMP(size, res, arg, this->m_mode, this->m_initial);
					break;
#endif
				default:
					this->CPU(size, res, arg, this->m_mode, this->m_initial);
					break;
				}
			}
			
			
#pragma mark - CPU
			template<typename OutIterator, typename InIterator>
			void CPU(size_t size, OutIterator res, InIterator arg, ScanMode mode, T initial);
			
			
#ifdef SKEPU_OPENMP
			template<typename OutIterator, typename InIterator>
			void OMP(size_t size, OutIterator res, InIterator arg, ScanMode mode, T initial);
			
#endif

#ifdef SKEPU_CUDA
			template<typename OutIterator, typename InIterator>
			void CU(size_t size, OutIterator res, InIterator arg, ScanMode mode, T initial);
		
			template<typename OutIterator, typename InIterator>
			void scanSingleThread_CU(size_t deviceID, size_t size, OutIterator res, InIterator arg, ScanMode mode, T initial);
			
			template<typename OutIterator, typename InIterator>
			void scanMulti_CU(size_t numDevices, size_t size, OutIterator res, InIterator arg, ScanMode mode, T initial);
				
			T scanLargeVectorRecursively_CU(size_t deviceID, DeviceMemPointer_CU<T>* input, DeviceMemPointer_CU<T>* output, const std::vector<DeviceMemPointer_CU<T>*>& blockSums, size_t numElements, ScanMode mode, T init, size_t level = 0);
			
			T scanLargeVectorRecursivelyM_CU(DeviceMemPointer_CU<T>* input, DeviceMemPointer_CU<T>* output, const std::vector<DeviceMemPointer_CU<T>*>& blockSums, size_t numElements, ScanMode mode, T init, Device_CU* device, size_t level = 0);
			
#endif
		
#ifdef SKEPU_OPENCL
			template<typename OutIterator, typename InIterator>
			void CL(size_t size, OutIterator res, InIterator arg, ScanMode mode, T initial);
			
			void scanLargeVectorRecursively_CL(DeviceMemPointer_CL<T>* res, size_t deviceID, DeviceMemPointer_CL<T>* input, DeviceMemPointer_CL<T>* output,
				const std::vector<DeviceMemPointer_CL<T>*>& blockSums, size_t numElements, ScanMode mode, T initial, size_t level = 0);
			
			template<typename OutIterator, typename InIterator>
			void scanSingle_CL(size_t deviceID, size_t size, OutIterator res, InIterator arg, ScanMode mode, T initial);
			
			template<typename OutIterator, typename InIterator>
			void scanNumDevices_CL(size_t numDevices, size_t size, OutIterator res, InIterator arg, ScanMode mode, T initial);
			
#endif
		
		}; // class Scan
	} // namespace backend
} // namespace skepu2


#include "impl/scan/scan_cpu.inl"
#include "impl/scan/scan_omp.inl"
#include "impl/scan/scan_cl.inl"
#include "impl/scan/scan_cu.inl"

#endif
