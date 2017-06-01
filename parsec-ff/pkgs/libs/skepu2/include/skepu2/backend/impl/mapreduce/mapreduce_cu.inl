/*! \file mapreduce_cu.inl
 *  \brief Contains the definitions of CUDA specific member functions for the MapReduce skeleton.
 */

#ifdef SKEPU_CUDA

#include <cuda.h>

#include "../../reduce_helpers.h"


namespace skepu2
{
	namespace backend
	{
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs> 
		typename ReduceFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::mapReduceSingleThread_CU(size_t deviceID, size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args)
		{
			// Setup parameters
			Device_CU *device = this->m_environment->m_devices_CU[deviceID];
			auto eArgs  = std::make_tuple(get<EI, CallArgs...>(args...)...);
			auto aArgs  = std::make_tuple(get<AI, CallArgs...>(args...)...);
			auto scArgs = std::make_tuple(get<CI, CallArgs...>(args...)...);
			
			// Number of threads per block, taken from NVIDIA source
			size_t numBlocks, numThreads;
			std::tie(numThreads, numBlocks) = getNumBlocksAndThreads(size, this->m_selected_spec->GPUBlocks(), this->m_selected_spec->GPUThreads());
			
			auto elwiseMemP = std::make_tuple(std::get<EI>(eArgs).getParent().updateDevice_CU(std::get<EI>(eArgs).getAddress(), size, deviceID, AccessMode::Read)...);
			auto anyMemP    = std::make_tuple(std::get<AI-arity>(aArgs).cudaProxy(deviceID, MapFunc::anyAccessMode[AI-arity])...);
			
			// Create the output memory
			DeviceMemPointer_CU<Ret> outMemP(&res, numBlocks, device);
			Ret *d_odata = outMemP.getDeviceDataPointer();
			
			// First map and reduce all elements blockwise so that each block produces one element. After this the mapping is complete
			const size_t sharedMemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(Ret) : numThreads * sizeof(Ret);
			
			DEBUG_TEXT_LEVEL1("CUDA MapReduce: numBlocks = " << numBlocks << ", numThreads = " << numThreads);
			
#ifdef USE_PINNED_MEMORY
			this->m_cuda_kernel<<<numBlocks, numThreads, sharedMemSize, device->m_streams[0]>>>
#else
			this->m_cuda_kernel<<<numBlocks, numThreads, sharedMemSize>>>
#endif
			(
				std::get<EI>(elwiseMemP)->getDeviceDataPointer()...,
				std::get<AI-arity>(anyMemP).second...,
				std::get<CI-arity-anyArity>(scArgs)...,
				d_odata,
				elwise_width(eArgs),
				size,
				0
			);
			
			size_t threads, blocks;
			for (size_t s = numBlocks; s > 1; s = (s + threads * 2 - 1) / (threads * 2))
			{
				std::tie(threads, blocks) = getNumBlocksAndThreads(s, this->m_selected_spec->GPUBlocks(), this->m_selected_spec->GPUThreads());
				
#ifdef USE_PINNED_MEMORY
				CallReduceKernel_WithStream(m_cuda_reduce_kernel, s, threads, blocks, d_odata, d_odata, device->m_streams[0]);
#else
				CallReduceKernel(this->m_cuda_reduce_kernel, s, threads, blocks, d_odata, d_odata);
#endif
			}
			
			// Make sure the data is marked as changed by the device
			pack_expand((std::get<AI-arity>(anyMemP).first->changeDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
			
			// Copy back result
			outMemP.changeDeviceData();
			outMemP.copyDeviceToHost(1);
			
			return res;
		}
		
		
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs> 
		typename ReduceFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::mapReduceMultiStream_CU(size_t deviceID, size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args)
		{
			CHECK_CUDA_ERROR(cudaSetDevice(deviceID));
			
			// Setup parameters
			Device_CU *device = this->m_environment->m_devices_CU[deviceID];
			const size_t numKernels = std::min<size_t>(device->getNoConcurrentKernels(), size);
			const size_t numElemPerSlice = size / numKernels;
			const size_t rest = size % numKernels;
			
			auto eArgs  = std::make_tuple(get<EI, CallArgs...>(args...)...);
			auto aArgs  = std::make_tuple(get<AI, CallArgs...>(args...)...);
			auto scArgs = std::make_tuple(get<CI, CallArgs...>(args...)...);
			
			Ret result[numKernels];
			typename to_device_pointer_cu<decltype(eArgs)>::type elwiseMemP[numKernels];
			typename to_proxy_cu<decltype(aArgs)>::type anyMemP[numKernels];
			DeviceMemPointer_CU<Ret>* outMemP[numKernels];
			
			size_t numBlocks[numKernels];
			size_t numThreads[numKernels];
			
			// First create CUDA memory if not created already.
			for (size_t i = 0; i < numKernels; ++i)
			{
				const size_t numElem = numElemPerSlice + ((i == numKernels-1) ? rest : 0);
				const size_t baseIndex = i * numElemPerSlice;
				std::tie(numThreads[i], numBlocks[i]) = getNumBlocksAndThreads(numElem, this->m_selected_spec->GPUBlocks(), this->m_selected_spec->GPUThreads());
				
				DEBUG_TEXT_LEVEL1("CUDA MapReduce: Kernel " << i << ", numElem = " << numElem << ", numBlocks = " << numBlocks[i] << ", numThreads = " << numThreads[i]);
				
				elwiseMemP[i] = std::make_tuple(std::get<EI>(eArgs).getParent().updateDevice_CU(std::get<EI>(eArgs).getAddress() + baseIndex, numElem, deviceID, AccessMode::None, false, i)...);
				anyMemP[i]    = std::make_tuple(std::get<AI-arity>(aArgs).cudaProxy(deviceID, AccessMode::None, false, i)...);
				outMemP[i] = new DeviceMemPointer_CU<Ret>(&result[i], numBlocks[i], m_environment->m_devices_CU.at(deviceID));
			}

			// Breadth-first memory transfers and kernel executions
			// First input memory transfer
			for (size_t i = 0; i < numKernels; ++i)
			{
				const size_t numElem = numElemPerSlice + ((i == numKernels-1) ? rest : 0);
				const size_t baseIndex = i * numElemPerSlice;
			
				elwiseMemP[i] = std::make_tuple(std::get<EI>(eArgs).getParent().updateDevice_CU(std::get<EI>(eArgs).getAddress() + baseIndex, numElem, deviceID, AccessMode::Read, false, i)...);
				anyMemP[i] = std::make_tuple(std::get<AI-arity>(aArgs).cudaProxy(deviceID, MapFunc::anyAccessMode[AI-arity], false, i)...);
			}

			// Kernel executions
			for (size_t i = 0; i < numKernels; ++i)
			{
				const size_t numElem = numElemPerSlice + ((i == numKernels-1) ? rest : 0);
				const size_t baseIndex = i * numElemPerSlice;
				const size_t sharedMemSize = (numThreads[i] <= 32) ? 2 * numThreads[i] * sizeof(Ret) : numThreads[i] * sizeof(Ret);
				Ret *deviceOutMemP = outMemP[i]->getDeviceDataPointer();

#ifdef USE_PINNED_MEMORY
				this->m_cuda_kernel<<<numBlocks[i], numThreads[i], sharedMemSize, device->m_streams[i]>>>
#else
				this->m_cuda_kernel<<<numBlocks[i], numThreads[i], sharedMemSize>>>
#endif
				(
					std::get<EI>(elwiseMemP[i])->getDeviceDataPointer()...,
					std::get<AI-arity>(anyMemP[i]).second...,
					std::get<CI-arity-anyArity>(scArgs)...,
					deviceOutMemP,
					elwise_width(eArgs),
					numElem,
					baseIndex
				);
				
				size_t threads, blocks;
				for (size_t s = numBlocks[i]; s > 1; s = (s + (threads*2-1)) / (threads*2))
				{
					std::tie(threads, blocks) = getNumBlocksAndThreads(s, this->m_selected_spec->GPUBlocks(), this->m_selected_spec->GPUThreads());

#ifdef USE_PINNED_MEMORY
					CallReduceKernel_WithStream(m_cuda_reduce_kernel, s, threads, blocks, deviceOutMemP, deviceOutMemP, stream);
#else
					CallReduceKernel(m_cuda_reduce_kernel, s, threads, blocks, deviceOutMemP, deviceOutMemP);
#endif
				}
				
				// Make sure the data is marked as changed by the device
				pack_expand((std::get<AI-arity>(anyMemP[i]).first->changeDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
				
				//Copy back result
				outMemP[i]->changeDeviceData();
			}
			
			// Joins the threads and reduces the results on the CPU, yielding the total result.
			for (size_t i = 0; i < numKernels; ++i)
			{
				outMemP[i]->copyDeviceToHost(1);
				res = ReduceFunc::CPU(res, result[i]);
				delete outMemP[i];
			}
			
			return res;
		}
		
		
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename ...CallArgs> 
		typename ReduceFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::mapReduceMultiStreamMultiGPU_CU(size_t useNumGPU, size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args)
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
				Device_CU *device = this->m_environment->m_devices_CU[i];
				CHECK_CUDA_ERROR(cudaSetDevice(i));
				numKernels[i] = std::min<size_t>(device->getNoConcurrentKernels(), numElemPerDevice);
				maxKernels = std::max(maxKernels, numKernels[i]);
				
				size_t temp = numElemPerDevice + ((i == useNumGPU-1) ? deviceRest : 0);
				numElemPerStream[i] = temp / numKernels[i];
				streamRest[i] = temp % numKernels[i];
			}
			
			Ret result[MAX_GPU_DEVICES][maxKernels];
			typename to_device_pointer_cu<decltype(eArgs)>::type elwiseMemP[MAX_GPU_DEVICES][maxKernels];
			typename to_proxy_cu<decltype(aArgs)>::type anyMemP[MAX_GPU_DEVICES][maxKernels];
			DeviceMemPointer_CU<Ret>* outMemP[MAX_GPU_DEVICES][maxKernels];
			
			size_t numBlocks[MAX_GPU_DEVICES][maxKernels];
			size_t numThreads[MAX_GPU_DEVICES][maxKernels];
			
			// First create CUDA memory if not created already.
			for (int i = 0; i < useNumGPU; ++i)
			{
				CHECK_CUDA_ERROR(cudaSetDevice(i));
				for (int j = 0; j < numKernels[i]; ++j)
				{
					const size_t numElem = numElemPerStream[i] + ((j == numKernels[i]-1) ? streamRest[i] : 0);
					const size_t baseIndex = i * numElemPerDevice + j * numElemPerStream[i];
					std::tie(numThreads[i][j], numBlocks[i][j]) = getNumBlocksAndThreads(numElem, this->m_selected_spec->GPUBlocks(), this->m_selected_spec->GPUThreads());
					
					DEBUG_TEXT_LEVEL1("CUDA MapReduce: Device " << i << ", kernel = " << j << "numElem = " << numElem << ", numBlocks = " << numBlocks[i][j] << ", numThreads = " << numThreads[i][j]);
					
					elwiseMemP[i][j] = std::make_tuple(std::get<EI>(eArgs).getParent().updateDevice_CU((std::get<EI>(eArgs) + baseIndex).getAddress(), numElem, i, AccessMode::None, false, j)...);
					anyMemP[i][j] = std::make_tuple(std::get<AI-arity>(aArgs).cudaProxy(i, AccessMode::None, false, j)...);
					outMemP[i][j] = new DeviceMemPointer_CU<Ret>(&result[i][j], numBlocks[i][j], m_environment->m_devices_CU.at(i));
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
					const size_t sharedMemSize = (numThreads[i][j] <= 32) ? 2 * numThreads[i][j] * sizeof(Ret) : numThreads[i][j] * sizeof(Ret);
					Ret *deviceOutMemP = outMemP[i][j]->getDeviceDataPointer();
					
					this->m_cuda_kernel<<<numBlocks[i][j], numThreads[i][j], sharedMemSize, this->m_environment->m_devices_CU[i]->m_streams[j]>>>
					(
						std::get<EI>(elwiseMemP[i][j])->getDeviceDataPointer()...,
						std::get<AI-arity>(anyMemP[i][j]).second...,
						std::get<CI-arity-anyArity>(scArgs)...,
						deviceOutMemP,
						elwise_width(eArgs), numElem, baseIndex
					);
					
					size_t threads, blocks;
					for (size_t s = numBlocks[i][j]; s > 1; s = (s + (threads*2-1)) / (threads*2))
					{
						std::tie(threads, blocks) = getNumBlocksAndThreads(s, maxBlocks, maxThreads);
						CallReduceKernel_WithStream(m_cuda_reduce_kernel, s, threads, blocks, deviceOutMemP, deviceOutMemP, stream);
					}
					
					// Make sure the data is marked as changed by the device
					pack_expand((std::get<AI-arity>(anyMemP[i][j]).second->changeDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
					
					// Copy back result
					outMemP[i][j]->changeDeviceData();
				}
			}
			
			// Joins the threads and reduces the results on the CPU, yielding the total result.
			for (size_t i = 0; i < useNumGPU; ++i)
			{
				for (size_t j = 0; j < numKernels[i]; ++j)
				{
					outMemP[i][j]->copyDeviceToHost(1);
					res = ReduceFunc::CPU(res, result[i][j]);
					delete outMemP[i][j];
				}
			}
#endif
			return res;
		}
		
		
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename... CallArgs> 
		typename ReduceFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::mapReduceSingleThreadMultiGPU_CU(size_t numDevices, size_t size, pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, Ret &res, CallArgs&&... args)
		{
			// Divide elements among participating devices
			const size_t numElemPerSlice = size / numDevices;
			const size_t rest = size % numDevices;
			
			auto eArgs = std::make_tuple(get<EI, CallArgs...>(args...)...);
			auto aArgs = std::make_tuple(get<AI, CallArgs...>(args...)...);
			auto scArgs = std::make_tuple(get<CI, CallArgs...>(args...)...);
			
			typename to_device_pointer_cu<decltype(eArgs)>::type elwiseMemP[MAX_GPU_DEVICES];
			typename to_proxy_cu<decltype(aArgs)>::type anyMemP[MAX_GPU_DEVICES];
			
			Ret result[MAX_GPU_DEVICES];
			DeviceMemPointer_CU<Ret>* outMemP[MAX_GPU_DEVICES];
			
			// Setup parameters
			size_t numThreads[MAX_GPU_DEVICES];
			size_t numBlocks[MAX_GPU_DEVICES];
			
			// First create CUDA memory if not created already.
			for (size_t i = 0; i < numDevices; ++i)
			{
				const size_t numElem = numElemPerSlice + ((i == numDevices-1) ? rest : 0);
				const size_t baseIndex = i * numElemPerSlice;
				std::tie(numThreads[i], numBlocks[i]) = getNumBlocksAndThreads(numElem, this->m_selected_spec->GPUBlocks(), this->m_selected_spec->GPUThreads());
				
				DEBUG_TEXT_LEVEL1("CUDA MapReduce: device " << i << ", numElem = " << numElem << ", numBlocks = " << numBlocks[i] << ", numThreads = " << numThreads[i]);
				
				elwiseMemP[i] = std::make_tuple(std::get<EI>(eArgs).getParent().updateDevice_CU((std::get<EI>(eArgs) + baseIndex).getAddress(), numElem, i, AccessMode::None)...);
				anyMemP[i]    = std::make_tuple(std::get<AI-arity>(aArgs).cudaProxy(i, AccessMode::None)...);
				outMemP[i] = new DeviceMemPointer_CU<Ret>(&result[i], numBlocks[i], this->m_environment->m_devices_CU[i]);
			}
			
			// Fill out argument struct with right information and start threads.
			for (size_t i = 0; i < numDevices; ++i)
			{
				CHECK_CUDA_ERROR(cudaSetDevice(i));
				const size_t numElem = numElemPerSlice + ((i == numDevices-1) ? rest : 0);
				const size_t baseIndex = i * numElemPerSlice;
				const size_t sharedMemSize = (numThreads[i] <= 32) ? 2 * numThreads[i] * sizeof(Ret) : numThreads[i] * sizeof(Ret);
				Ret *deviceOutMemP = outMemP[i]->getDeviceDataPointer();
				
				// Copies the elements to the device
				elwiseMemP[i] = std::make_tuple(std::get<EI>(eArgs).getParent().updateDevice_CU((std::get<EI>(eArgs) + baseIndex).getAddress(), numElem, i, AccessMode::Read)...);
				anyMemP[i] = std::make_tuple(std::get<AI-arity>(aArgs).cudaProxy(i, MapFunc::anyAccessMode[AI-arity])...);
				
				// First map and reduce all elements blockwise so that each block produces one element. After this the mapping is complete
#ifdef USE_PINNED_MEMORY
				this->m_cuda_kernel<<<numBlocks[i], numThreads[i], sharedMemSize, this->m_environment->m_devices_CU.at(i)->m_streams[0]>>>
#else
				this->m_cuda_kernel<<<numBlocks[i], numThreads[i], sharedMemSize>>>
#endif
				(
					std::get<EI>(elwiseMemP[i])->getDeviceDataPointer()...,
					std::get<AI-arity>(anyMemP[i]).second...,
					std::get<CI-arity-anyArity>(scArgs)...,
					deviceOutMemP,
					elwise_width(eArgs),
					numElem,
					baseIndex
				);
				
				size_t threads, blocks;
				for (size_t s = numBlocks[i]; s > 1; s = (s + threads * 2 - 1) / (threads * 2))
				{
					std::tie(threads, blocks) = getNumBlocksAndThreads(s, this->m_selected_spec->GPUBlocks(), this->m_selected_spec->GPUThreads());

#ifdef USE_PINNED_MEMORY
					CallReduceKernel_WithStream(m_cuda_reduce_kernel, s, threads, blocks, deviceOutMemP, deviceOutMemP, stream);
#else
					CallReduceKernel(m_cuda_reduce_kernel, s, threads, blocks, deviceOutMemP, deviceOutMemP);
#endif
				}
				
				// Make sure the data is marked as changed by the device
				pack_expand((std::get<AI-arity>(anyMemP[i]).first->changeDeviceData(hasWriteAccess(MapFunc::anyAccessMode[AI-arity])), 0)...);
				outMemP[i]->changeDeviceData();
			}
			
			// Joins the threads and reduces the results on the CPU, yielding the total result.
			for (size_t i = 0; i < numDevices; ++i)
			{
				outMemP[i]->copyDeviceToHost(1);
				res = ReduceFunc::CPU(res, result[i]);
				delete outMemP[i];
			}
			
			CHECK_CUDA_ERROR(cudaSetDevice(m_environment->bestCUDADevID));
		
			return res;
		}
		
		
		template<size_t arity, typename MapFunc, typename ReduceFunc, typename CUDAKernel, typename CUDAReduceKernel, typename CLKernel>
		template<size_t... EI, size_t... AI, size_t... CI, typename... CallArgs> 
		typename ReduceFunc::Ret MapReduce<arity, MapFunc, ReduceFunc, CUDAKernel, CUDAReduceKernel, CLKernel>
		::CUDA(size_t size, pack_indices<EI...> ei, pack_indices<AI...> ai, pack_indices<CI...> ci, Ret &res, CallArgs&&... args)
		{
			DEBUG_TEXT_LEVEL1("CUDA MapReduce: size = " << size << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads());
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CU.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			if (numDevices <= 1)
			{
#ifdef USE_PINNED_MEMORY
				
				//Checks whether or not the GPU supports MemoryTransfer/KernelExec overlapping, if not call mapReduceSingleThread function
				if (this->m_environment->m_devices_CU.at(this->m_environment->bestCUDADevID)->isOverlapSupported())
					return mapReduceMultiStream_CU(this->m_environment->bestCUDADevID, size, ei, ai, ci, res, args...);
				
#endif
				return mapReduceSingleThread_CU(this->m_environment->bestCUDADevID, size, ei, ai, ci, res, args...);
			}
			
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
#ifdef USE_PINNED_MEMORY
			
			// if pinned memory is used but the device does not support overlap the function continues with the previous implementation.
			// if the multistream version is being used the function will exit at this point.
			if (this->m_environment->supportsCUDAOverlap())
				return mapReduceMultiStreamMultiGPU_CU(numDevices, size, ei, ai, ci, res, args...);
			
#endif
			return mapReduceSingleThreadMultiGPU_CU(numDevices, size, ei, ai, ci, res, args...);
		}
	} // namespace backend
} // namespace skepu2

#endif
