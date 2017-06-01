/*! \file call_cu.inl
 *  \brief Contains the definitions of CUDA specific member functions for the Call skeleton.
 */

#ifdef SKEPU_CUDA

#include <cuda.h>

namespace skepu2
{
	namespace backend
	{
		template<typename CallFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs> 
		void Call<CallFunc, CUDAKernel, CLKernel>
		::callSingleThread_CU(size_t deviceID, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto aArgs = std::make_tuple(get<AI, CallArgs...>(args...)...);
			auto scArgs = std::make_tuple(get<CI, CallArgs...>(args...)...);
			
			// Setup parameters
			const size_t numThreads = this->m_selected_spec->GPUThreads();
			const size_t numBlocks = this->m_selected_spec->GPUBlocks();
			
			DEBUG_TEXT_LEVEL1("CUDA Call: numBlocks = " << numBlocks << ", numThreads = " << numThreads);
			
			// Copies the elements to the device
			auto anyMemP = std::make_tuple(std::get<AI>(aArgs).cudaProxy(deviceID, CallFunc::anyAccessMode[AI])...);
			
			// Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
			this->m_cuda_kernel<<<numBlocks, numThreads, 0, this->m_environment->m_devices_CU.at(deviceID)->m_streams[0]>>>
#else
			this->m_cuda_kernel<<<numBlocks, numThreads>>>
#endif // USE_PINNED_MEMORY
			(
				std::get<AI>(anyMemP).second...,
				std::get<CI-anyArity>(scArgs)...
			);
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<AI>(anyMemP).first->changeDeviceData(hasWriteAccess(CallFunc::anyAccessMode[AI])), 0)...);
			
#ifdef TUNER_MODE
			cudaDeviceSynchronize();
#endif // TUNER_MODE
		}
		
		
		template<typename CallFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... AI, size_t... CI, typename ...CallArgs> 
		void Call<CallFunc, CUDAKernel, CLKernel>
		::callMultiStream_CU(size_t deviceID, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			CHECK_CUDA_ERROR(cudaSetDevice(deviceID));
			size_t numKernels = this->m_environment->m_devices_CU.at(deviceID)->getNoConcurrentKernels();
			
			auto aArgs = std::make_tuple(get<AI, CallArgs...>(args...)...);
			auto scArgs = std::make_tuple(get<CI, CallArgs...>(args...)...);
			
			typename to_proxy_cu<decltype(aArgs)>::type anyMemP[numKernels];
			
			// First create CUDA memory if not created already.
			for (size_t i = 0; i < numKernels; ++i)
				anyMemP[i] = std::make_tuple(std::get<AI>(aArgs).getParent().cudaProxy( deviceID, AccessMode::None, false, i)...);
			
			// Breadth-first memory transfers and kernel executions
			// First input memory transfer
			for (size_t i = 0; i < numKernels; ++i)
				anyMemP[i] = std::make_tuple(std::get<AI>(aArgs).getParent().cudaProxy(deviceID, CallFunc::anyAccessMode[AI], false, i)...);
			
			// Kernel executions
			for(size_t i = 0; i < numKernels; ++i)
			{
				const size_t numThreads = this->m_selected_spec->GPUThreads();
				const size_t numBlocks = this->m_selected_spec->GPUBlocks();
				
				DEBUG_TEXT_LEVEL1("CUDA Call: Kernel " << i << ", numElem = " << numElem << ", numBlocks = " << numBlocks << ", numThreads = " << numThreads);
				
#ifdef USE_PINNED_MEMORY
				this->m_cuda_kernel<<<numBlocks, numThreads, 0, this->m_environment->m_devices_CU.at(deviceID)->m_streams[i]>>>
#else
				this->m_cuda_kernel<<<numBlocks, numThreads>>>
#endif // USE_PINNED_MEMORY
				(
					std::get<AI>(anyMemP[i]).second...,
					std::get<CI-anyArity>(scArgs)...
				);
				
				// Change device data
				pack_expand((std::get<AI>(anyMemP[i]).first->changeDeviceData(hasWriteAccess(CallFunc::anyAccessMode[AI])), 0)...);
			}

#ifdef TUNER_MODE
			cudaDeviceSynchronize();
#endif // TUNER_MODE
		}
		
		
		template<typename CallFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... AI, size_t... CI, typename ...CallArgs> 
		void Call<CallFunc, CUDAKernel, CLKernel>
		::callMultiStreamMultiGPU_CU(size_t useNumGPU, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
#ifdef USE_PINNED_MEMORY
			size_t numKernels[MAX_GPU_DEVICES];
			size_t numElemPerStream[MAX_GPU_DEVICES];
			size_t streamRest[MAX_GPU_DEVICES];
			size_t maxKernels = 0;
			
			auto aArgs = std::make_tuple(get<AI, CallArgs...>(args...)...);
			auto scArgs = std::make_tuple(get<CI, CallArgs...>(args...)...);
			
			for (size_t i = 0; i < useNumGPU; ++i)
			{
				CHECK_CUDA_ERROR(cudaSetDevice(i));
				numKernels[i] = std::min<size_t>(this->m_environment->m_devices_CU.at(i)->getNoConcurrentKernels(), numElemPerDevice);
				maxKernels = std::max(maxKernels, numKernels[i]);
				
				size_t temp = numElemPerDevice + ((i == useNumGPU-1) ? deviceRest : 0);
				numElemPerStream[i] = temp / numKernels[i];
				streamRest[i] = temp % numKernels[i];
			}
			
			typename to_proxy_cu<decltype(aArgs)>::type anyMemP[MAX_GPU_DEVICES][maxKernels];
			
			// First create CUDA memory if not created already.
			for (size_t i = 0; i < useNumGPU; ++i)
			{
				CHECK_CUDA_ERROR(cudaSetDevice(i));
				for (size_t j = 0; j < numKernels[i]; ++j)
					anyMemP[i][j] = std::make_tuple(std::get<AI>(aArgs).cudaProxy(i, AccessMode::None, false, j)...);
			}
			
			// First input memory transfer
			for (size_t i = 0; i < useNumGPU; ++i)
			{
				CHECK_CUDA_ERROR(cudaSetDevice(i));
				for (size_t j = 0; j < numKernels[i]; ++j)
					anyMemP[i][j] = std::make_tuple(std::get<AI>(aArgs).cudaProxy(i, CallFunc::anyAccessMode[AI], false, j)...);
			}
			
			// Kernel executions
			for (size_t i = 0; i < useNumGPU; ++i)
			{
				CHECK_CUDA_ERROR(cudaSetDevice(i));
				for (size_t j = 0; j < numKernels[i]; ++j)
				{
					const size_t numThreads = std::min(this->m_selected_spec->GPUThreads(), numElem);
					const size_t numBlocks = std::max<size_t>(1, std::min( (numElem / numThreads + (numElem % numThreads == 0 ? 0:1)), this->m_selected_spec->GPUBlocks()));
					
					DEBUG_TEXT_LEVEL1("CUDA Call: Device " << i << ", kernel = " << j << "numElem = " << numElem << ", numBlocks = " << numBlocks << ", numThreads = " << numThreads);
					
					this->m_cuda_kernel<<<numBlocks, numThreads, 0, this->m_environment->m_devices_CU.at(i)->m_streams[j]>>>
					(
						std::get<AI>(anyMemP[i][j]).second...,
						std::get<CI-anyArity>(scArgs)...
					);
					
					pack_expand((std::get<AI>(anyMemP[i][j]).first->changeDeviceData(hasWriteAccess(CallFunc::anyAccessMode[AI])), 0)...);
				}
			}
#endif // USE_PINNED_MEMORY
		}
		
		
		template<typename CallFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs> 
		void Call<CallFunc, CUDAKernel, CLKernel>
		::callSingleThreadMultiGPU_CU(size_t numDevices, pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			auto aArgs = std::make_tuple(get<AI, CallArgs...>(args...)...);
			auto scArgs = std::make_tuple(get<CI, CallArgs...>(args...)...);
			
			typename to_proxy_cu<decltype(aArgs)>::type anyMemP[MAX_GPU_DEVICES];
			
			// First create CUDA memory if not created already.
			for (size_t i = 0; i < numDevices; ++i)
				anyMemP[i] = std::make_tuple(std::get<AI>(aArgs).cudaProxy(i, AccessMode::None)...);
			
			// Fill out argument struct with right information and start threads.
			for (size_t i = 0; i < numDevices; ++i)
			{
				CHECK_CUDA_ERROR(cudaSetDevice(i));
				const size_t numThreads = this->m_selected_spec->GPUThreads();
				const size_t numBlocks = this->m_selected_spec->GPUBlocks();
				
				DEBUG_TEXT_LEVEL1("CUDA Call: device " << i << ", numBlocks = " << numBlocks << ", numThreads = " << numThreads);
				
				anyMemP[i] = std::make_tuple(std::get<AI>(aArgs).cudaProxy(i, CallFunc::anyAccessMode[AI])...);
				
				// Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
				this->m_cuda_kernel<<<numBlocks, numThreads, 0, this->m_environment->m_devices_CU.at(i)->m_streams[0]>>>
#else
				this->m_cuda_kernel<<<numBlocks, numThreads>>>
#endif // USE_PINNED_MEMORY
				(
					std::get<AI>(anyMemP[i]).second...,
					std::get<CI-anyArity>(scArgs)...
				);
				
				// Change device data
				pack_expand((std::get<AI>(anyMemP[i]).first->changeDeviceData(hasWriteAccess(CallFunc::anyAccessMode[AI])), 0)...);
			}
			
			CHECK_CUDA_ERROR(cudaSetDevice(this->m_environment->bestCUDADevID));
		}
		
		
		template<typename CallFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... AI, size_t... CI, typename... CallArgs> 
		void Call<CallFunc, CUDAKernel, CLKernel>
		::CUDA(pack_indices<AI...> ai, pack_indices<CI...> ci, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("CUDA Call: Devices = " << this->m_selected_spec->devices()
				<< ", blocks = " << this->m_selected_spec->GPUBlocks() << ", threads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CU.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			if (numDevices <= 1)
			{
#ifdef USE_PINNED_MEMORY
				
				// Checks whether or not the GPU supports MemoryTransfer/KernelExec overlapping, if not call callSingleThread function
				if (this->m_environment->m_devices_CU.at(this->m_environment->bestCUDADevID)->isOverlapSupported())
					return callMultiStream_CU(this->m_environment->bestCUDADevID, ai, ci, args...);
				
#endif // USE_PINNED_MEMORY
				
				return callSingleThread_CU(this->m_environment->bestCUDADevID, ai, ci, args...);
			}
			
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
#ifdef USE_PINNED_MEMORY
			
			// if pinned memory is used but the device does not support overlap the function continues with the previous implementation.
			// if the multistream version is being used the function will exit at this point.
			if (this->m_environment->supportsCUDAOverlap())
				return callMultiStreamMultiGPU_CU(numDevices, ai, ci, args...);
			
#endif // USE_PINNED_MEMORY
			
			return callSingleThreadMultiGPU_CU(numDevices, ai, ci, args...);
		}
	} // namespace backend
} // namespace skepu2

#endif
