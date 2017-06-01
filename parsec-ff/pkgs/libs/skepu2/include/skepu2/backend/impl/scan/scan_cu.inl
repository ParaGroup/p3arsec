/*! \file scan_cu.inl
 *  \brief Contains the definitions of CUDA specific member functions for the Scan skeleton.
 */

#ifdef SKEPU_CUDA

#include <cuda.h>
#include <iostream>

namespace skepu2
{
	namespace backend
	{
		/*!
		 *  Scans a Vector using the same recursive algorithm as NVIDIA SDK. First the vector is scanned producing partial results for each block.
		 *  Then the function is called recursively to scan these partial results, which in turn can produce partial results and so on.
		 *  This continues until only one block with partial results is left. Used by multi-GPU CUDA implementation.
		 */
		template <typename ScanFunc, typename CUDAScan, typename CUDAScanUpdate, typename CUDAScanAdd, typename CLKernel>
		typename ScanFunc::Ret Scan<ScanFunc, CUDAScan, CUDAScanUpdate, CUDAScanAdd, CLKernel>
		::scanLargeVectorRecursivelyM_CU(DeviceMemPointer_CU<T>* input, DeviceMemPointer_CU<T>* output, const std::vector<DeviceMemPointer_CU<T>*>& blockSums, size_t numElements, ScanMode mode, T init, Device_CU* device, size_t level)
		{
			const unsigned int deviceID = device->getDeviceID();
			const size_t numThreads = device->getMaxThreads();
			const size_t maxBlocks = device->getMaxBlocks();
			const size_t numBlocks = std::min(numElements / numThreads + (numElements % numThreads == 0 ? 0:1), maxBlocks);
			const size_t totalNumBlocks = numElements / numThreads + (numElements % numThreads == 0 ? 0:1);
			const size_t sharedMemSize = sizeof(T) * numThreads * 2;
			const size_t updateSharedMemSize = sizeof(T) * numThreads;
			const int isInclusive = (mode == ScanMode::Inclusive || level > 0) ? 1 : 0;
			T ret = 0;
			
			DeviceMemPointer_CU<T> ret_mem_p(&ret, 1, Environment<int>::getInstance()->m_devices_CU.at(deviceID));
			
#ifdef USE_PINNED_MEMORY
			this->m_cuda_scan_kernel<<< numBlocks, numThreads, sharedMemSize, this->m_environment->m_devices_CU.at(deviceID)->m_streams[0] >>>
				(input->getDeviceDataPointer(), output->getDeviceDataPointer(), blockSums[level]->getDeviceDataPointer(), numThreads, numElements);
#else
			this->m_cuda_scan_kernel<<< numBlocks, numThreads, sharedMemSize >>>
				(input->getDeviceDataPointer(), output->getDeviceDataPointer(), blockSums[level]->getDeviceDataPointer(), numThreads, numElements);
#endif
			if (numBlocks > 1)
				this->scanLargeVectorRecursivelyM_CU(blockSums[level], blockSums[level], blockSums, totalNumBlocks, mode, init, device, level+1);
			
#ifdef USE_PINNED_MEMORY
			this->m_cuda_scan_update_kernel<<< numBlocks, numThreads, updateSharedMemSize, this->m_environment->m_devices_CU.at(deviceID)->m_streams[0] >>>
				(output->getDeviceDataPointer(), blockSums[level]->getDeviceDataPointer(), isInclusive, init, numElements, ret_mem_p.getDeviceDataPointer());
#else
			this->m_cuda_scan_update_kernel<<< numBlocks, numThreads, updateSharedMemSize >>>
				(output->getDeviceDataPointer(), blockSums[level]->getDeviceDataPointer(), isInclusive, init, numElements, ret_mem_p.getDeviceDataPointer());
#endif
			
			ret_mem_p.changeDeviceData();
			ret_mem_p.copyDeviceToHost();
			
			return ret;
		}
		
		
		/*!
		 *  Scans a Vector using the same recursive algorithm as NVIDIA SDK. First the vector is scanned producing partial results for each block.
		 *  Then the function is called recursively to scan these partial results, which in turn can produce partial results and so on.
		 *  This continues until only one block with partial results is left.
		 */
		template<typename ScanFunc, typename CUDAScan, typename CUDAScanUpdate, typename CUDAScanAdd, typename CLKernel>
		typename ScanFunc::Ret Scan<ScanFunc, CUDAScan, CUDAScanUpdate, CUDAScanAdd, CLKernel>
		::scanLargeVectorRecursively_CU(size_t deviceID, DeviceMemPointer_CU<T>* input, DeviceMemPointer_CU<T>* output, const std::vector<DeviceMemPointer_CU<T>*>& blockSums, size_t size, ScanMode mode, T init, size_t level)
		{
			const size_t numThreads = this->m_selected_spec->GPUThreads();
			const size_t numBlocks = std::min(size / numThreads + (size % numThreads == 0 ? 0:1), this->m_selected_spec->GPUBlocks());
			const size_t sharedMemSize = sizeof(T) * numThreads * 2;
			const size_t updateSharedMemSize = sizeof(T) * numThreads;
			const int isInclusive = (mode == ScanMode::Inclusive || level > 0) ? 1 : 0;
			T ret = 0;
			
			DEBUG_TEXT_LEVEL1("CUDA Scan: level = " << level << ", size = " << size << ", numBlocks = " << numBlocks << ", numThreads = " << numThreads << "\n");
			
			// Return value used for multi-GPU
			DeviceMemPointer_CU<T> ret_mem_p(&ret, 1, m_environment->m_devices_CU.at(deviceID));
			
			this->m_cuda_scan_kernel<<< numBlocks, numThreads, sharedMemSize >>>
				(input->getDeviceDataPointer(), output->getDeviceDataPointer(), blockSums[level]->getDeviceDataPointer(), numThreads, size);
			
			if (numBlocks > 1)
			{
				const size_t totalNumBlocks = size / numThreads + (size % numThreads == 0 ? 0:1);
				this->scanLargeVectorRecursively_CU(deviceID, blockSums[level], blockSums[level], blockSums, totalNumBlocks, mode, init, level+1);
			}
				
			this->m_cuda_scan_update_kernel<<< numBlocks, numThreads, updateSharedMemSize >>>
				(output->getDeviceDataPointer(), blockSums[level]->getDeviceDataPointer(), isInclusive, init, size, ret_mem_p.getDeviceDataPointer());
			
			return ret;
		}
		
		
		template<typename T>
		std::vector<DeviceMemPointer_CU<T>*> allocateBlockSums(size_t size, size_t numThreads, Device_CU *device)
		{
			size_t numEl = size;
			size_t numBlocks;
			std::vector<DeviceMemPointer_CU<T>*> blockSums;
			
			do
			{
				numBlocks = numEl / numThreads + (numEl % numThreads == 0 ? 0:1);
				if (numBlocks >= 1)
				{
					blockSums.push_back(new DeviceMemPointer_CU<T>(NULL, numBlocks, device));
				}
				numEl = numBlocks;
			}
			while (numEl > 1);
			
			return blockSums;
		}
		
		
		template<typename ScanFunc, typename CUDAScan, typename CUDAScanUpdate, typename CUDAScanAdd, typename CLKernel>
		template<typename OutIterator, typename InIterator>
		void Scan<ScanFunc, CUDAScan, CUDAScanUpdate, CUDAScanAdd, CLKernel>
		::scanMulti_CU(size_t numDevices, size_t size, OutIterator res, InIterator arg, ScanMode mode, T initial)
		{
			const size_t numElemPerSlice = size / numDevices;
			const size_t rest = size % numDevices;
			
			Vector<T> deviceSums;
			T ret = 0;
			
			// Fill out argument struct with right information and start threads.
			for (size_t i = 0; i < numDevices; ++i)
			{
				const size_t numElem = numElemPerSlice + ((i == numDevices-1) ? rest : 0);
				const size_t numThreads = this->m_selected_spec->GPUThreads();
				const std::vector<DeviceMemPointer_CU<T>*> blockSums = allocateBlockSums<T>(size, numThreads, this->m_environment->m_devices_CU.at(i));;
				
				typename InIterator::device_pointer_type_cu in_mem_p = arg.getParent().updateDevice_CU((arg + i * numElemPerSlice).getAddress(), numElem, i, AccessMode::Read);
				typename OutIterator::device_pointer_type_cu out_mem_p = res.getParent().updateDevice_CU((res + i * numElemPerSlice).getAddress(), numElem, i, AccessMode::Write);
				
				cudaSetDevice(i);
				ret = scanLargeVectorRecursivelyM_CU(in_mem_p, out_mem_p, blockSums, numElem, mode, initial, this->m_environment->m_devices_CU.at(i));
				deviceSums.push_back(ret);
				out_mem_p->changeDeviceData();
				
				// Clean up
				for (size_t i = 0; i < blockSums.size(); ++i)
					delete blockSums[i];
			}
			
			CPU(deviceSums.size(), deviceSums.begin(), deviceSums.begin(), ScanMode::Inclusive, T{});
			
			for (size_t i = 1; i < numDevices; ++i)
			{
				const size_t numElements = numElemPerSlice + ((i == numDevices-1) ? rest : 0);
				const size_t numThreads = this->m_selected_spec->GPUThreads();
				const size_t numBlocks = std::min(numElements / numThreads + (numElements % numThreads == 0 ? 0:1), this->m_selected_spec->GPUBlocks());
				
				DEBUG_TEXT_LEVEL1("CUDA Scan (add device sums): device " << i << ", numElem = " << numElements << ", numBlocks = " << numBlocks << ", numThreads = " << numThreads << "\n");
				
				typename OutIterator::device_pointer_type_cu out_mem_p = res.getParent().updateDevice_CU((res + i * numElemPerSlice).getAddress(), numElements, i, AccessMode::ReadWrite);
				cudaSetDevice(i);
				
#ifdef USE_PINNED_MEMORY
				this->m_cuda_scan_add_kernel<<< numBlocks, numThreads, 0, (this->m_environment->m_devices_CU.at(i)->m_streams[0]) >>>
					(out_mem_p->getDeviceDataPointer(), deviceSums(i-1), numElements);
#else
				this->m_cuda_scan_add_kernel<<< numBlocks, numThreads >>>
					(out_mem_p->getDeviceDataPointer(), deviceSums(i-1), numElements);
#endif
			}
			cudaSetDevice(m_environment->bestCUDADevID);
		}
		
		
		/*!
		 *  Performs the Scan on an input range using \em CUDA with a separate output range. Used when scanning the array on
		 *  one device using one host thread. Allocates space for intermediate results from each block, and then calls scanLargeVectorRecursively_CU.
		 */
		template<typename ScanFunc, typename CUDAScan, typename CUDAScanUpdate, typename CUDAScanAdd, typename CLKernel>
		template<typename OutIterator, typename InIterator>
		void Scan<ScanFunc, CUDAScan, CUDAScanUpdate, CUDAScanAdd, CLKernel>
		::scanSingleThread_CU(size_t deviceID, size_t size, OutIterator res, InIterator arg, ScanMode mode, T initial)
		{
			// Setup parameters
			const size_t numThreads = this->m_selected_spec->GPUThreads();
			const std::vector<DeviceMemPointer_CU<T>*> blockSums = allocateBlockSums<T>(size, numThreads, this->m_environment->m_devices_CU.at(deviceID));
			
			typename InIterator::device_pointer_type_cu in_mem_p = arg.getParent().updateDevice_CU(arg.getAddress(), size, deviceID, AccessMode::Read);
			typename OutIterator::device_pointer_type_cu out_mem_p = res.getParent().updateDevice_CU(res.getAddress(), size, deviceID, AccessMode::Write);
			
			this->scanLargeVectorRecursively_CU(deviceID, in_mem_p, out_mem_p, blockSums, size, mode, initial);
			
			// Clean up
			for (size_t i = 0; i < blockSums.size(); ++i)
				delete blockSums[i];
			
			out_mem_p->changeDeviceData();
		}
		
		
		/*!
		 *  Performs the Scan on an input range using \em CUDA with a separate output range. The function decides whether to perform
		 *  the scan on one device, calling scanSingleThread_CU or
		 *  on multiple devices, dividing the work between multiple devices.
		 */
		template<typename ScanFunc, typename CUDAScan, typename CUDAScanUpdate, typename CUDAScanAdd, typename CLKernel>
		template<typename OutIterator, typename InIterator>
		void Scan<ScanFunc, CUDAScan, CUDAScanUpdate, CUDAScanAdd, CLKernel>
		::CU(size_t size, OutIterator res, InIterator arg, ScanMode mode, T initial)
		{
			DEBUG_TEXT_LEVEL1("CUDA Scan: size = " << size << ", maxDevices = " << this->m_selected_spec->devices()
				<< ", maxBlocks = " << this->m_selected_spec->GPUBlocks() << ", maxThreads = " << this->m_selected_spec->GPUThreads() << "\n");
			
			const size_t numDevices = std::min(this->m_selected_spec->devices(), this->m_environment->m_devices_CU.size());
			
#ifndef SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
			if (numDevices <= 1)
				this->scanSingleThread_CU(this->m_environment->bestCUDADevID, size, res, arg, mode, initial);
			else
			
#endif // SKEPU_DEBUG_FORCE_MULTI_GPU_IMPL
			
				this->scanMulti_CU(numDevices, size, res, arg, mode, initial);
		}
	}
}

#endif

