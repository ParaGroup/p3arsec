/*! \file environment.inl
 *  \brief Contains member function definitions for the Environment class.
 */

#include <cstdlib>
#include <cstring>
#include <cassert>

#include "../bandwidthMeasure.h"
//#include "mapoverlap_convol_kernels.h"

namespace skepu2
{
	namespace backend
	{
		/*!
		 *  Static member initialization, keeps track of number of created instances.
		 */
		template <typename T>
		Environment<T>* Environment<T>::instance = 0;
		
		
		template <typename T>
		bool Environment<T>::getGroupMapping(int groupId, Backend::Type &type)
		{
			if (m_cacheGroupResult.first == groupId)
			{
				type = m_cacheGroupResult.second;
				return true;
			}
			for (int i=0; i<m_groupMapping.size(); ++i)
			{
				if (m_groupMapping[i].first == groupId)
				{
					m_cacheGroupResult = m_groupMapping[i];
					type = m_cacheGroupResult.second;
					return true;
				}
			}
			return false;
		}
		
		
		template <typename T>
		void Environment<T>::addGroupMapping(int groupId, Backend::Type type)
		{
			m_cacheGroupResult = std::make_pair(groupId, type);
			m_groupMapping.push_back(std::make_pair(groupId, type));
		}
		
		
		template<typename T>
		void Environment<T>::clearGroupMapping()
		{
			m_groupMapping.clear();
		}
		
		
		/*!
		 *  Gets pointer to first instance, at first call a new instance is created.
		 *
		 *  \return Pointer to class instance.
		 */
		template <typename T>
		Environment<T>* Environment<T>::getInstance()
		{
			if(instance == 0)
			{
				instance = new Environment;
				_destroyer.SetEnvironment(instance);
			}
			return instance;
		}
		
		
		/*!
		 *  The constructor initializes the devices.
		 */
		template <typename T>
		Environment<T>::Environment()
		{
			m_cacheGroupResult.first = -1;
			init();
			
			m_numDevices = 0;
#ifdef SKEPU_CUDA   
			m_peerAccessEnabled = 0; // by default not enabled...
#endif
		
#ifdef SKEPU_OPENCL
			init_CL();
			createOpenCLProgramForMatrixTranspose();
#endif
		
		
#ifdef SKEPU_CUDA
			init_CU();
		//	CHECK_CUDA_ERROR(cudaFuncSetCacheConfig("conv_cuda_shared_kernel", cudaFuncCachePreferShared));
		//	assert(m_devices_CU.size() == 1);
			assert(SKEPU_CUDA_DEV_ID == bestCUDADevID);
			bwDataStruct = measureOrLoadCUDABandwidth(bestCUDADevID, false);
#else
			//	std::cerr << "[SkePU Warning]: bwDataStruct is not initialized as CUDA is not enabled...\n";
			bwDataStruct.cpu_id = -1;
			bwDataStruct.timing_htd = -1;
			bwDataStruct.latency_htd = -1;
			bwDataStruct.timing_dth = -1;
			bwDataStruct.latency_dth = -1;
			bwDataStruct.timing_dtd = -1;
			bwDataStruct.latency_dtd = -1;
#endif
		}
		
		
		/*!
		 *  The constructor initializes the devices.
		 */
		template <typename T>
		Environment<T>::~Environment()
		{
			DEBUG_TEXT_LEVEL1("Environment destructor");
			
#ifdef SKEPU_OPENCL
			if (getDataTypeCL<T>() == "int") // to ensure that only we create devices when using Environment<int> and rest we just copy information from Environment<int>
			{
				for (Device_CL *device : this->m_devices_CL)
					delete device;
			}
#endif // SKEPU_OPENCL
		
#ifdef SKEPU_CUDA
			// disable peer-peer memcopy if enabled for different gpu combinations...
			for (int i=0; i<m_peerCopyGpuIDsVector.size(); ++i)
			{
				DEBUG_TEXT_LEVEL1("Disabling peer-peer access between GPU " << m_peerCopyGpuIDsVector[i].first << " <--> " << m_peerCopyGpuIDsVector[i].second << "\n");
				CHECK_CUDA_ERROR(cudaSetDevice(m_peerCopyGpuIDsVector[i].first));
				CHECK_CUDA_ERROR(cudaDeviceDisablePeerAccess(m_peerCopyGpuIDsVector[i].second));
				CHECK_CUDA_ERROR(cudaSetDevice(m_peerCopyGpuIDsVector[i].second));
				CHECK_CUDA_ERROR(cudaDeviceDisablePeerAccess(m_peerCopyGpuIDsVector[i].first));
			}
			
			finishAll_CU();
			
			for (Device_CU *device : this->m_devices_CU)
				delete device;
			
#endif // SKEPU_CUDA
		}
		
		
		/*!
		 *  General initialization.
		 */
		template <typename T>
		void Environment<T>::init()
		{
		//	srand(time(0));
		}
		
#ifdef SKEPU_CUDA
		
		inline bool IsAppBuiltAs64()
		{
#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
			return 1;
#else
			return 0;
#endif
		}
		
		
		/*!
		 * A helper function to check if peer-to-peer mem transfers can be done bwteen 2 gpus.. if yes, it enables it and return true, otherwise false....
		 */
		inline bool cudaPeerToPeerMemAccess(int gpuId1, int gpuId2)
		{
			if(IsAppBuiltAs64() == false)
				return false; // can only be enabled if app is build for 64-bit OS...
			
			bool enable = false;
			
			// Query using the CUDA device properties API
			cudaDeviceProp prop1, prop2;
			
			// 1 and 2 here are the numbers of the GPUs
			CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop1, gpuId1));
			CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop2, gpuId2));
			
#ifdef _WIN32
			if(prop1.tccDriver != 1 || prop2.tccDriver != 1)
				return false;
#else
			if(prop1.major < 2 || prop2.major < 2)
				return false;
#endif
			
			int access2from1 = -1;
			int access1from2 = -1;
			int flags = 0;
			
			CHECK_CUDA_ERROR(cudaDeviceCanAccessPeer(&access2from1, gpuId1, gpuId2));
			CHECK_CUDA_ERROR(cudaDeviceCanAccessPeer(&access1from2, gpuId2, gpuId1));
			
			bool sameComplex = false;
			if (access2from1==1 && access1from2==1)
			{
				sameComplex = true;
			}
			
		//	if(compatibleDriver)
		//		std::cerr << "compatibleDriver\n";
		//	if(sameComplex)
		//		std::cerr << "sameComplex\n";
			
			if (sameComplex)
			{
				// Enable peer access
				CHECK_CUDA_ERROR(cudaSetDevice(gpuId1));
				CHECK_CUDA_ERROR(cudaDeviceEnablePeerAccess(gpuId2,flags)); //flags=0
				CHECK_CUDA_ERROR(cudaSetDevice(gpuId2));
				CHECK_CUDA_ERROR(cudaDeviceEnablePeerAccess(gpuId1,flags)); //flags=0
				
				//	std::cerr << "compatibleDriver and sameComplex exist HURRAHHHH!!!!!\n";
				
				int nbytes = 1024;
				// Allocate some data
				float *gpu1data, *gpu2data;
				CHECK_CUDA_ERROR(cudaSetDevice(gpuId1));
				CHECK_CUDA_ERROR(cudaMalloc(&gpu1data, nbytes));
				CHECK_CUDA_ERROR(cudaSetDevice(gpuId2));
				CHECK_CUDA_ERROR(cudaMalloc(&gpu2data, nbytes));
				
				// Do the p2p copy!
				if(cudaSuccess == cudaMemcpyPeer(gpu1data, gpuId1, gpu2data, gpuId2, nbytes))
				{
					DEBUG_TEXT_LEVEL1("%%%% NOTICE: Peer-to-Peer copy between GPU " << gpuId1 << " - GPU " << gpuId2 << " is enabled !!!!\n");
					enable = true;
				}
			}
			return enable;
		}
		
		
		/*!
		 *  Initializes the CUDA devices available. Also does a warm-up call to bind host thread to device.
		 */
		template <typename T>
		void Environment<T>::init_CU()
		{
			cudaGetLastError(); // clear any previous errors, if any
			cudaError_t err;
			int numDevices = 0;
			
			// Create devices available
			err = cudaGetDeviceCount(&numDevices);
			
			m_numDevices = numDevices;
			if (m_numDevices <= 0)
				SKEPU_ERROR("No SKEPU_CUDA enabled devices found!\n");
			
			if (err != cudaSuccess)
				SKEPU_ERROR("cudaGetDeviceCount failed!\n");
			
			// if more devices, found, we used only what is specified as MAX_GPU_DEVICES.
			if (m_numDevices > MAX_GPU_DEVICES)
				m_numDevices = MAX_GPU_DEVICES;
			
			int best_SM_arch = 0;
			int sm_per_multiproc = 0, major = 0;
			Device_CU *device;
			
			for (int i = 0; i < m_numDevices; ++i)
			{
				device=new Device_CU(i);
				m_devices_CU.push_back(device);
				major = device->getMajorVersion();
				
				if (major > 0 && major < 9999)
				{
					best_SM_arch = MAX(best_SM_arch, major);
				}
				
				CHECK_CUDA_ERROR(cudaSetDevice(device->getDeviceID()));
				
				int up = 0;
				DeviceMemPointer_CU<int> warm(&up, 1, device);
				
				cudaGetLastError();
			}
			
			int max_compute_perf = 0;
			
#ifndef SKEPU_CUDA_DEV_ID   
			int max_perf_device  = 0;
#endif   
			
			for(int i=0; i<m_numDevices; i++)
			{
				device=m_devices_CU.at(i);
				
				major = device->getMajorVersion();
				sm_per_multiproc = device->getSmPerMultiProc();
				
				int compute_perf  = device->getNumComputeUnits() * sm_per_multiproc * device->getClockRate();
				
				if( compute_perf  > max_compute_perf )
				{
					// If we find GPU with SM major > 2, search only these
					if ( best_SM_arch > 2 )
					{
						// If our device==dest_SM_arch, choose this, or else pass
						if (major == best_SM_arch)
						{
							max_compute_perf  = compute_perf;
#ifndef SKEPU_CUDA_DEV_ID               
							max_perf_device   = i;
#endif               
						}
					}
					else
					{
						max_compute_perf  = compute_perf;
#ifndef SKEPU_CUDA_DEV_ID
						max_perf_device   = i;
#endif            
					}
				}
			}
			
			/*! code to check if peer-peer memory transfers between 2 gpus are possible using GPUDirectin CUDA 4.0 or above */
			if (m_numDevices > 1)
			{
#ifdef USE_PINNED_MEMORY
				
				bool allEnabled = true;
				for (int i=0; i<m_numDevices; ++i)
				{
					for (int j=i+1; j<m_numDevices; ++j) /*! j=i+1 and j=0 because if peer access for "i,j" is same as "j,i") */
					{
						if (cudaPeerToPeerMemAccess(i, j))
							m_peerCopyGpuIDsVector.push_back(std::make_pair(i,j));
						else
							allEnabled = false;
					}
				}
				
				if (allEnabled)
					m_peerAccessEnabled = 1; // enabled for all...
				else if (m_peerCopyGpuIDsVector.empty())
					m_peerAccessEnabled = 0; // not enabled for any...
				else
					m_peerAccessEnabled = -1; // enabled for some of them...
				     
			//	m_peerAccessEnabled = 0;  
#endif      
			}
			
#ifdef SKEPU_CUDA_DEV_ID
			bestCUDADevID = SKEPU_CUDA_DEV_ID; // Set user specified index
#else
			bestCUDADevID = max_perf_device; // Set it as the best CUDA device
#endif
			
			if (m_numDevices > 0)
			{
				CHECK_CUDA_ERROR(cudaSetDevice(bestCUDADevID)); // Set the best CUDA device for execution
			}
		}
		
		
		template <typename T>
		bool Environment<T>::supportsCUDAOverlap()
		{
			for (size_t i = 0; i < m_numDevices; ++i)
			{
				if (!this->m_devices_CU.at(i)->isOverlapSupported())
					return false;
			}
			return true;
		}
		
		
		/*!
		 *  Finish all CUDA functions on all devices. Optionally can specify a range of IDs to block for.
		 *
		 * \param lowID optional. specifies the lowest CUDA ID to do synchronization on.
		 * \param highID optional. specifies the highest CUDA ID to do synchronization on.
		 */
template <typename T>
		void Environment<T>::finishAll_CU(int lowID, int highID)
		{
			assert(m_numDevices == m_devices_CU.size());
			
			if (SKEPU_UNLIKELY(m_numDevices == 0))
				return;
			
			if (lowID < 0)
				lowID = 0;
			if (highID < 1 || highID > m_numDevices)
				highID = m_numDevices;
			
			for (int i=lowID; i < highID; i++)
			{
			//	DEBUG_TEXT_LEVEL1("%%%%%\n** Synchronizing device "<<i << "\n%%%%%%%%%%%%%%%%%\n");
				/*CHECK_CUDA_ERROR(*/cudaSetDevice(i)/*)*/;
			//	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
				cudaDeviceSynchronize();
			}
			
			/*CHECK_CUDA_ERROR(*/cudaSetDevice(bestCUDADevID)/*)*/;
		}
#endif // SKEPU_CUDA
		
		
#ifdef SKEPU_OPENCL
		
		/*!
		 *  Initializes the OpenCL devices available.
		 */
		template <typename T>
		void Environment<T>::init_CL()
		{
			if (getDataTypeCL<T>() == "int") // to ensure that only we create devices when using Environment<int> and rest we just copy information from Environment<int>
			{
				//Create platforms- Uptil now, there could be two more common platforms that we target that can resides in single machine (NVIDIA, ATI)
				cl_platform_id temp_platforms[2];
				cl_uint no_platforms;
				cl_int err = clGetPlatformIDs(2, temp_platforms, &no_platforms);
				CL_CHECK_ERROR(err, "Error getting OpenCL platform list");
				
				int platform_ind=0;
				if(no_platforms > 1) // for more than platform, by default look for nvidia, ifdef ATI then look for ATI, else use whatever platform is available
				{
					for(; platform_ind <no_platforms; platform_ind++)
					{
						char inf[1024];
						cl_platform_info temp_info;
						err= clGetPlatformInfo(temp_platforms[platform_ind], CL_PLATFORM_VENDOR, sizeof(char)*1024, inf, NULL);
						
#ifdef USE_ATI_OPENCL
						if (!strcmp(inf, "Advanced Micro Devices, Inc."))
							break;
#else
						if (!strcmp(inf, "NVIDIA Corporation"))
							break;
#endif
					}
				}
				
				if (platform_ind == no_platforms) // should not occur in normal situations.
					SKEPU_ERROR("ERROR! No platform is selected");
				
				//Get number of devices in system (either CPU or GPU)
				cl_uint numDevices;
				clGetDeviceIDs(temp_platforms[platform_ind], /*CL_DEVICE_TYPE_CPU |*/ CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, 0, NULL, &numDevices);
				
				if (numDevices == 0)
					SKEPU_ERROR("No SKEPU_OPENCL devices found!\n");
				
				// if more devices, found, we used only what is specified as MAX_GPU_DEVICES.
				if (numDevices > MAX_GPU_DEVICES)
					numDevices = MAX_GPU_DEVICES;
				
				//Specify platform props
				cl_context_properties props[3];
				props[0] = (cl_context_properties)CL_CONTEXT_PLATFORM;
				props[1] = (cl_context_properties)temp_platforms[platform_ind];
				props[2] = (cl_context_properties)0;
				
				//Create and get those devices found
				cl_device_id deviceList[MAX_GPU_DEVICES];
				err = clGetDeviceIDs(temp_platforms[platform_ind], /*CL_DEVICE_TYPE_CPU |*/ CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR, numDevices, deviceList, &numDevices);
				CL_CHECK_ERROR(err, "Error getting OpenCL devices");
				
				cl_context context = clCreateContext(0, numDevices, deviceList, NULL, NULL, &err);
				CL_CHECK_ERROR(err, "Error creating OpenCL context");
				
				for (cl_uint i = 0; i < numDevices; ++i)
				{
					cl_device_type temptype;
					clGetDeviceInfo(deviceList[i], CL_DEVICE_TYPE, sizeof(temptype), &temptype, NULL); // get device type
					
					m_devices_CL.push_back(new Device_CL(deviceList[i], temptype, context));
					
#if SKEPU_DEBUG > 0
					Device_CL* d = m_devices_CL.back();
					std::cout << "OpenCL device " << i << " properties:\n\t"
						<< "maxBlocks: "<< d->getMaxBlocks() << "\n\t"
						<< "maxThreads: "<< d->getMaxThreads() << "\n\t"
						<< "sharedMemoryPerBlock: "<<d->getSharedMemPerBlock() << "\n\t"
						<< "getMaxBlockSize: " << d->getMaxBlockSize() << "\n\t"
						<< "getNumComputeUnits: "<< d->getNumComputeUnits() << "\n\t"
						<< "getGlobalMemSize: "<<d->getGlobalMemSize() << "\n\t"
						<< "getType: "<< d->getType() << "\n\n";
#endif
				}
				m_numDevices = numDevices;
			}
			else // Just copy it from Environment<int>::getInstance() to ensure not to invoke it multiple times.
			{
				Environment<int> *envInt = Environment<int>::getInstance();
				this->m_numDevices = envInt->m_devices_CL.size();
				for (int i=0; i < m_numDevices; i++)
					this->m_devices_CL.push_back(envInt->m_devices_CL.at(i)); // = envInt->m_devices_CL; // Just copy it
			}
		}
		
		
		/*!
		 *  Finish all OpenCL functions on all devices.
		 */
		template <typename T>
		void Environment<T>::finishAll_CL()
		{
			for(std::vector<Device_CL*>::iterator it = m_devices_CL.begin(); it != m_devices_CL.end(); ++it)
			{
				clFinish((*it)->getQueue());
			}
		}
#endif // SKEPU_OPENCL
		
		
		/*!
		 *  Wrapper for CUDA and OpenCL variants. Does not do anything if neither is used. Makes code more portable.
		 */
		template <typename T>
		void Environment<T>::finishAll()
		{
#ifdef SKEPU_OPENCL
			finishAll_CL();
#endif
#ifdef SKEPU_CUDA
			finishAll_CU();
#endif
		}
		
		
		
#ifdef SKEPU_OPENCL
		
		/*!
		 *
		 *  OpenCL Transpose kernel. Modified the transpose kernel provided by NVIDIA to make it work for any problem size rather than just perfect size such as 1024X1024.
		 */
		static const std::string TransposeKernelNoBankConflicts_CL = R"~~~(
		__kernel void transposeNoBankConflicts(__global TYPE* odata, __global TYPE* idata, size_t width, size_t height, __local TYPE* sdata)
		{
			int xIndex = get_group_id(0) * TILE_DIM + get_local_id(0);
			int yIndex = get_group_id(1) * TILE_DIM + get_local_id(1);
			int index_in = xIndex + (yIndex) * width;
			
			if (xIndex < width && yIndex < height)
				sdata[get_local_id(1) * TILE_DIM + get_local_id(0)] = idata[index_in];
			
			xIndex = get_group_id(1) * TILE_DIM + get_local_id(0);
			yIndex = get_group_id(0) * TILE_DIM + get_local_id(1);
			int index_out = xIndex + (yIndex) * height;
			
			barrier(CLK_LOCAL_MEM_FENCE);
			
			if (xIndex < height && yIndex < width)
				odata[index_out] = sdata[get_local_id(0) * TILE_DIM + get_local_id(1)];
		}
		)~~~";
		
		
#define TILE_DIM    16
#define BLOCK_ROWS  16
		
		/*!
		 *  A function called by the constructor. It creates the OpenCL program for the Matrix Transpose and saves a handle for
		 *  the kernel. The program is built from a string containing the above mentioned generic transpose kernel. The type
		 *  and function names in the generic kernel are relpaced by specific code before it is compiled by the OpenCL JIT compiler.
		 *
		 *  Also handles the use of doubles automatically by including "#pragma OPENCL EXTENSION cl_khr_fp64: enable" if doubles
		 *  are used.
		 */
		template <typename T>
		void Environment<T>::createOpenCLProgramForMatrixTranspose()
		{
			std::string kernelName = "transposeNoBankConflicts";
			std::string datatype_CL = getDataTypeCL<T>();
			std::stringstream totalSource;
			
			if (datatype_CL == "double")
				totalSource << "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";
			
			totalSource << cl_helpers::replaceSizeT(TransposeKernelNoBankConflicts_CL);
			
			std::stringstream buildOptions;
			buildOptions << "-DTYPE=\"" << datatype_CL << "\" -DTILE_DIM=" << TILE_DIM;
			
			DEBUG_TEXT_LEVEL3("Transpose kernel source:\n" << totalSource.str());
			DEBUG_TEXT_LEVEL3("Transpose kernel build options:\n" << buildOptions.str());
			
			// Builds the code and creates kernel for all devices
			for (Device_CL *device : this->m_devices_CL)
			{
				cl_int err;
				cl_program program = cl_helpers::buildProgram(device, totalSource.str(), buildOptions.str());
				cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &err);
				CL_CHECK_ERROR(err, "Error creating kernel '" << kernelName << "':" << err);
				this->m_transposeKernels_CL.emplace_back(kernel, device);
			}
		}
		
#endif // SKEPU_OPENCL
		
	} // end namespace backend
}// end namespace skepu2
