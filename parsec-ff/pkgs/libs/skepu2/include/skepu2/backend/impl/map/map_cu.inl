/*! \file map_cu.inl
 *  \brief Contains the definitions of CUDA specific member functions for the Map skeleton.
 */

#ifdef SKEPU_CUDA

#include <cuda.h>
#include <iostream>
#include <functional>

namespace skepu2
{
	namespace backend
	{
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename Iterator, typename ...CallArgs> 
		void Map<arity, MapFunc, CUDAKernel, CLKernel>
		::mapSingleThread_CU(size_t deviceID, size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args)
		{
			auto eArgs = std::make_tuple(get<EI, CallArgs...>(args...)...);
			auto aArgs = std::make_tuple(get<AI, CallArgs...>(args...)...);
			auto scArgs = std::make_tuple(get<CI, CallArgs...>(args...)...);
			
			// Setup parameters
			const size_t numThreads = std::min(this->m_selected_spec->GPUThreads(), size);
			const size_t numBlocks = std::max<size_t>(1, std::min( (size / numThreads + (size % numThreads == 0 ? 0:1)), this->m_selected_spec->GPUBlocks()));
			
			DEBUG_TEXT_LEVEL1("CUDA Map: numBlocks = " << numBlocks << ", numThreads = " << numThreads);
			
			// Copies the elements to the device
			auto elwiseMemP = std::make_tuple(std::get<EI>(eArgs).getParent().updateDevice_CU(std::get<EI>(eArgs).getAddress(), size, deviceID, AccessMode::Read)...);
			auto anyMemP = std::make_tuple(std::get<AI-arity>(aArgs).getParent().cudaProxy(deviceID, MapFunc::anyAccessMode[AI-arity])...);
			auto outMemP = res.getParent().updateDevice_CU(res.getAddress(), size, deviceID, AccessMode::Write);
			
			// Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
			this->m_cuda_kernel<<<numBlocks, numThreads, 0, this->m_environment->m_devices_CU.at(deviceID)->m_streams[0]>>>
#else
			this->m_cuda_kernel<<<numBlocks, numThreads>>>
#endif // USE_PINNED_MEMORY
			(
				std::get<EI>(elwiseMemP)->getDeviceDataPointer()...,
				std::get<AI-arity>(anyMemP).second...,
				std::get<CI-arity-anyArity>(scArgs)...,
				outMemP->getDeviceDataPointer(),
				res.getParent().total_cols(),
				size,
				0
			);
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<AI-arity>(anyMemP).first->changeDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			outMemP->changeDeviceData();
			
#ifdef TUNER_MODE
			cudaDeviceSynchronize();
#endif // TUNER_MODE
		}
		
		
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename Iterator, typename ...CallArgs> 
		void Map<arity, MapFunc, CUDAKernel, CLKernel>
		::mapMultiStream_CU(size_t deviceID, size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args)
		{
			CHECK_CUDA_ERROR(cudaSetDevice(deviceID));
			size_t numKernels = std::min<size_t>(this->m_environment->m_devices_CU.at(deviceID)->getNoConcurrentKernels(), size);
			size_t numElemPerSlice = size / numKernels;
			size_t rest = size % numKernels;
			
			auto eArgs = std::make_tuple(get<EI, CallArgs...>(args...)...);
			auto aArgs = std::make_tuple(get<AI, CallArgs...>(args...)...);
			auto scArgs = std::make_tuple(get<CI, CallArgs...>(args...)...);
			
			typename to_device_pointer_cu<decltype(eArgs)>::type elwiseMemP[numKernels];
			typename to_proxy_cu<decltype(aArgs)>::type anyMemP[numKernels];
			typename Iterator::device_pointer_type_cu outMemP[numKernels];
			
			// First create CUDA memory if not created already.
			for (size_t i = 0; i < numKernels; ++i)
			{
				const size_t numElem = numElemPerSlice + ((i == numKernels-1) ? rest : 0);
				const size_t baseIndex = i * numElemPerSlice;
				
				elwiseMemP[i] = std::make_tuple(std::get<EI>(eArgs).getParent().updateDevice_CU((std::get<EI>(eArgs) + baseIndex).getAddress(), numElem, deviceID, AccessMode::None, false, i)...);
				anyMemP[i] = std::make_tuple(std::get<AI-arity>(aArgs).cudaProxy(deviceID, AccessMode::None, false, i)...);
				outMemP[i] = res.getParent().updateDevice_CU((res + baseIndex).getAddress(), numElem, deviceID, AccessMode::None, false, i);
			}
			
			// Breadth-first memory transfers and kernel executions
			// First input memory transfer
			for (size_t i = 0; i < numKernels; ++i)
			{
				const size_t numElem = numElemPerSlice + ((i == numKernels-1) ? rest : 0);
				const size_t baseIndex = i * numElemPerSlice;
			
				elwiseMemP[i] = std::make_tuple(std::get<EI>(eArgs).getParent().updateDevice_CU((std::get<EI>(eArgs) + baseIndex).getAddress(), numElem, deviceID, AccessMode::Read, false, i)...);
				anyMemP[i] = std::make_tuple(std::get<AI-arity>(aArgs).getParent().cudaProxy(deviceID, MapFunc::anyAccessMode[AI-arity], false, i)...);
				outMemP[i] = res.getParent().updateDevice_CU((res + baseIndex).getAddress(), numElem, deviceID, AccessMode::Write, false, i);
			}
			
			// Kernel executions
			for(size_t i = 0; i < numKernels; ++i)
			{
				const size_t numElem = numElemPerSlice + ((i == numKernels-1) ? rest : 0);
				const size_t baseIndex = i * numElemPerSlice;
				const size_t numThreads = std::min(this->m_selected_spec->GPUThreads(), numElem);
				const size_t numBlocks = std::min(numElem / numThreads + (numElem % numThreads == 0 ? 0:1), this->m_selected_spec->GPUBlocks());
				
				DEBUG_TEXT_LEVEL1("CUDA Map: Kernel " << i << ", numElem = " << numElem << ", numBlocks = " << numBlocks << ", numThreads = " << numThreads);
				
#ifdef USE_PINNED_MEMORY
				this->m_cuda_kernel<<<numBlocks, numThreads, 0, this->m_environment->m_devices_CU.at(deviceID)->m_streams[i]>>>
#else
				this->m_cuda_kernel<<<numBlocks, numThreads>>>
#endif // USE_PINNED_MEMORY
				(
					std::get<EI>(elwiseMemP[i])->getDeviceDataPointer()...,
					std::get<AI-arity>(anyMemP[i]).second...,
					std::get<CI-arity-anyArity>(scArgs)...,
					outMemP[i]->getDeviceDataPointer(),
					res.getParent().total_cols(),
					numElem,
					baseIndex
				);
				
				// Change device data
				pack_expand((std::get<AI-arity>(anyMemP[i]).first->changeDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
				outMemP[i]->changeDeviceData();
			}

#ifdef TUNER_MODE
			cudaDeviceSynchronize();
#endif // TUNER_MODE
		}
		
		
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename Iterator, typename ...CallArgs> 
		void Map<arity, MapFunc, CUDAKernel, CLKernel>
		::mapMultiStreamMultiGPU_CU(size_t useNumGPU, size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args)
		{
#ifdef USE_PINNED_MEMORY
			const size_t numElemPerDevice = size / useNumGPU;
			const size_t deviceRest = size % useNumGPU;
			size_t numKernels[MAX_GPU_DEVICES];
			size_t numElemPerStream[MAX_GPU_DEVICES];
			size_t streamRest[MAX_GPU_DEVICES];
			size_t maxKernels = 0;
			
			auto eArgs = std::make_tuple(get<EI, CallArgs...>(args...)...);
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
			
			typename to_device_pointer_cu<decltype(eArgs)>::type elwiseMemP[MAX_GPU_DEVICES][maxKernels];
			typename to_proxy_cu<decltype(aArgs)>::type anyMemP[MAX_GPU_DEVICES][maxKernels];
			typename Iterator::device_pointer_type_cu outMemP[MAX_GPU_DEVICES][maxKernels];
			
			// First create CUDA memory if not created already.	
			for (size_t i = 0; i < useNumGPU; ++i)
			{
				CHECK_CUDA_ERROR(cudaSetDevice(i));
				for (size_t j = 0; j < numKernels[i]; ++j)
				{
					const size_t numElem = numElemPerStream[i] + ((j == numKernels[i]-1) ? streamRest[i] : 0);
					const size_t baseIndex = i * numElemPerDevice + j * numElemPerStream[i];
					
					elwiseMemP[i][j] = std::make_tuple(std::get<EI>(eArgs).getParent().updateDevice_CU((std::get<EI>(eArgs) + baseIndex).getAddress(), numElem, i, AccessMode::None, false, j)...);
					anyMemP[i][j]    = std::make_tuple(std::get<AI-arity>(aArgs).cudaProxy(i, AccessMode::None, false, j)...);
					outMemP[i][j]    = res.getParent().updateDevice_CU((res + baseIndex).getAddress(), numElem, i, AccessMode::None, false, j);
				}
			}
			
			// First input memory transfer
			for (size_t i = 0; i < useNumGPU; ++i)
			{
				CHECK_CUDA_ERROR(cudaSetDevice(i));
				for (size_t j = 0; j < numKernels[i]; ++j)
				{
					const size_t numElem = numElemPerStream[i] + ((j == numKernels[i]-1) ? streamRest[i] : 0);
					const size_t baseIndex = i * numElemPerDevice + j * numElemPerStream[i];
					
					elwiseMemP[i][j] = std::make_tuple(std::get<EI>(eArgs).getParent().updateDevice_CU((std::get<EI>(eArgs) + baseIndex).getAddress(), numElem, i, AccessMode::Read, false, j)...);
					anyMemP[i][j]    = std::make_tuple(std::get<AI-arity>(aArgs).cudaProxy(i, MapFunc::anyAccessMode[AI-arity], false, j)...);
					outMemP[i][j]    = res.getParent().updateDevice_CU((res +baseIndex).getAddress(), numElem, i, AccessMode::Write, false, j);
				}
			}
			
			// Kernel executions
			for (size_t i = 0; i < useNumGPU; ++i)
			{
				CHECK_CUDA_ERROR(cudaSetDevice(i));
				for (size_t j = 0; j < numKernels[i]; ++j)
				{
					const size_t numElem = numElemPerStream[i] + ((j == numKernels[i]-1) ? streamRest[i] : 0);
					const size_t baseIndex = i * numElemPerDevice + j * numElemPerStream[i];
					const size_t numThreads = std::min(this->m_selected_spec->GPUThreads(), numElem);
					const size_t numBlocks = std::max<size_t>(1, std::min( (numElem / numThreads + (numElem % numThreads == 0 ? 0:1)), this->m_selected_spec->GPUBlocks()));
					
					DEBUG_TEXT_LEVEL1("CUDA Map: Device " << i << ", kernel = " << j << "numElem = " << numElem << ", numBlocks = " << numBlocks << ", numThreads = " << numThreads);
					
					this->m_cuda_kernel<<<numBlocks, numThreads, 0, this->m_environment->m_devices_CU.at(i)->m_streams[j]>>>(
						std::get<EI>(elwiseMemP[i][j])->getDeviceDataPointer()...,
						std::get<AI-arity>(anyMemP[i][j]).second...,
						std::get<CI-arity-anyArity>(scArgs)...,
						outMemP[i][j]->getDeviceDataPointer(),
						res.getParent().total_cols(),
						numElem,
						baseIndex
					);
					
					pack_expand((std::get<AI-arity>(anyMemP[i][j]).first->changeDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
					outMemP[i][j]->changeDeviceData();
				}
			}
#endif // USE_PINNED_MEMORY
		}
		
		
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename Iterator, typename ...CallArgs> 
		void Map<arity, MapFunc, CUDAKernel, CLKernel>
		::mapSingleThreadMultiGPU_CU(size_t numDevices, size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Iterator res, CallArgs&&... args)
		{
			const size_t numElemPerSlice = size / numDevices;
			const size_t rest = size % numDevices;
			
			auto eArgs = std::make_tuple(get<EI, CallArgs...>(args...)...);
			auto aArgs = std::make_tuple(get<AI, CallArgs...>(args...)...);
			auto scArgs = std::make_tuple(get<CI, CallArgs...>(args...)...);
			
			typename to_device_pointer_cu<decltype(eArgs)>::type elwiseMemP[MAX_GPU_DEVICES];
			typename to_proxy_cu<decltype(aArgs)>::type anyMemP[MAX_GPU_DEVICES];
			typename Iterator::device_pointer_type_cu outMemP[MAX_GPU_DEVICES];
			
			// First create CUDA memory if not created already.
			for (size_t i = 0; i < numDevices; ++i)
			{
				const size_t numElem = numElemPerSlice + ((i == numDevices-1) ? rest : 0);
				const size_t baseIndex = i * numElemPerSlice;
				
				elwiseMemP[i] = std::make_tuple(std::get<EI>(eArgs).getParent().updateDevice_CU((std::get<EI>(eArgs) + baseIndex).getAddress(), numElem, i, AccessMode::None)...);
				anyMemP[i]    = std::make_tuple(std::get<AI-arity>(aArgs).cudaProxy(i, AccessMode::None)...);
				outMemP[i]    = res.getParent().updateDevice_CU((res + baseIndex).getAddress(), numElem, i, AccessMode::None);
			}
			
			// Fill out argument struct with right information and start threads.
			for (size_t i = 0; i < numDevices; ++i)
			{
				CHECK_CUDA_ERROR(cudaSetDevice(i));
				const size_t numElem = numElemPerSlice + ((i == numDevices-1) ? rest : 0);
				const size_t baseIndex = i * numElemPerSlice;
				const size_t numThreads = std::min(this->m_selected_spec->GPUThreads(), numElem);
				const size_t numBlocks = std::max<size_t>(1, std::min( (numElem / numThreads + (numElem % numThreads == 0 ? 0:1)), this->m_selected_spec->GPUBlocks()));
				
				DEBUG_TEXT_LEVEL1("CUDA Map: device " << i << ", numElem = " << numElem << ", numBlocks = " << numBlocks << ", numThreads = " << numThreads);
				
				elwiseMemP[i] = std::make_tuple(std::get<EI>(eArgs).getParent().updateDevice_CU((std::get<EI>(eArgs) + baseIndex).getAddress(), numElem, i, AccessMode::Read)...);
				anyMemP[i]    = std::make_tuple(std::get<AI-arity>(aArgs).cudaProxy(i, MapFunc::anyAccessMode[AI-arity])...);
				outMemP[i]    = res.getParent().updateDevice_CU((res + baseIndex).getAddress(), numElem, i, AccessMode::Write, true);
				
				// Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
				this->m_cuda_kernel<<<numBlocks, numThreads, 0, this->m_environment->m_devices_CU.at(i)->m_streams[0]>>>
#else
				this->m_cuda_kernel<<<numBlocks, numThreads>>>
#endif // USE_PINNED_MEMORY
				(
					std::get<EI>(elwiseMemP[i])->getDeviceDataPointer()...,
					std::get<AI-arity>(anyMemP[i]).second...,
					std::get<CI-arity-anyArity>(scArgs)...,
					outMemP[i]->getDeviceDataPointer(),
					res.getParent().total_cols(),
					numElem,
					baseIndex
				);
				
				// Change device data
				pack_expand((std::get<AI-arity>(anyMemP[i]).first->changeDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
				outMemP[i]->changeDeviceData();
			}
			
			CHECK_CUDA_ERROR(cudaSetDevice(m_environment->bestCUDADevID));
			
			res.getParent().setValidFlag(false);
		}
		
		
		template<size_t arity, typename MapFunc, typename CUDAKernel, typename CLKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename Iterator, typename... CallArgs> 
		void Map<arity, MapFunc, CUDAKernel, CLKernel>
		::CUDA(size_t size, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, Iterator res, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("CUDA Map: size = " << size << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CU.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			if (numDevices <= 1)
			{
#ifdef USE_PINNED_MEMORY
				
				// Checks whether or not the GPU supports MemoryTransfer/KernelExec overlapping, if not call mapSingleThread function
				if (this->m_environment->m_devices_CU.at(m_environment->bestCUDADevID)->isOverlapSupported())
					this->mapMultiStream_CU(this->m_environment->bestCUDADevID, size, ei, ai, ci, res, args...);
				
#endif // USE_PINNED_MEMORY
				
				this->mapSingleThread_CU(this->m_environment->bestCUDADevID, size, ei, ai, ci, res, args...);
			}
			
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
#ifdef USE_PINNED_MEMORY
			
			// if pinned memory is used but the device does not support overlap the function continues with the previous implementation.
			// if the multistream version is being used the function will exit at this point.
			if (this->m_environment->supportsCUDAOverlap())
				this->mapMultiStreamMultiGPU_CU(numDevices, size, ei, ai, ci, res, args...);
			
#endif // USE_PINNED_MEMORY
			
			this->mapSingleThreadMultiGPU_CU(numDevices, size, ei, ai, ci, res, args...);
		}
	} // namespace backend
} // namespace skepu2

#endif
