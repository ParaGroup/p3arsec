/*! \file device_cl.h
*  \brief Contains a class declaration for the object that represents an OpenCL device.
 */

#ifndef DEVICE_CL_H
#define DEVICE_CL_H

#ifdef SKEPU_OPENCL

#include <iostream>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace skepu2
{
	namespace backend
	{
		/*!
		 *  \ingroup helpers
		 */
		/*!
		 *  \struct openclGenProp
		 *
		 *  A helper struct to openclDeviceProp. Used to help with the fetching of properties. See Device_CL.
		 */
		struct openclGenProp
		{
			cl_device_info param_name;
			size_t param_value_size;
			void* param_value;
		};
		
		/*!
		 *  \ingroup helpers
		 */
		/*!
		 *  \struct openclDeviceProp
		 *
		 *  A struct used to store OpenCL device properties. Adds the neccessary properties to a list which
		 *  can be used to fetch all those properties in a for-loop. See Device_CL.
		 */
		struct openclDeviceProp
		{
			openclDeviceProp()
			{
				openclGenProp temp;
				
				temp.param_name = CL_DEVICE_ADDRESS_BITS;
				temp.param_value_size = sizeof(cl_uint);
				temp.param_value = &DEVICE_ADDRESS_BITS;
				propertyList.push_back(temp);
				
				temp.param_name = CL_DEVICE_MAX_WORK_GROUP_SIZE;
				temp.param_value_size = sizeof(size_t);
				temp.param_value = &DEVICE_MAX_WORK_GROUP_SIZE;
				propertyList.push_back(temp);
				
				temp.param_name = CL_DEVICE_MAX_COMPUTE_UNITS;
				temp.param_value_size = sizeof(cl_uint);
				temp.param_value = &DEVICE_MAX_COMPUTE_UNITS;
				propertyList.push_back(temp);
				
				temp.param_name = CL_DEVICE_GLOBAL_MEM_SIZE;
				temp.param_value_size = sizeof(cl_ulong);
				temp.param_value = &DEVICE_GLOBAL_MEM_SIZE;
				propertyList.push_back(temp);
				
				temp.param_name = CL_DEVICE_LOCAL_MEM_SIZE;
				temp.param_value_size = sizeof(cl_ulong);
				temp.param_value = &DEVICE_LOCAL_MEM_SIZE;
				propertyList.push_back(temp);
			}
			
			std::vector<openclGenProp> propertyList;
			
			cl_uint DEVICE_ADDRESS_BITS;
			cl_bool DEVICE_AVAILABLE;
			cl_bool DEVICE_COMPILER_AVAILABLE;
			cl_device_fp_config DEVICE_DOUBLE_FP_CONFIG;
			cl_bool DEVICE_ENDIAN_LITTLE;
			cl_bool DEVICE_ERROR_CORRECTION_SUPPORT;
			cl_device_exec_capabilities DEVICE_EXECUTION_CAPABILITIES;
			char* DEVICE_EXTENSIONS;
			cl_ulong DEVICE_GLOBAL_MEM_CACHE_SIZE;
			cl_device_mem_cache_type DEVICE_GLOBAL_MEM_CACHE_TYPE;
			cl_uint DEVICE_GLOBAL_MEM_CACHELINE_SIZE;
			cl_ulong DEVICE_GLOBAL_MEM_SIZE;
			cl_device_fp_config DEVICE_HALF_FP_CONFIG;
			cl_bool DEVICE_IMAGE_SUPPORT;
			size_t DEVICE_IMAGE2D_MAX_HEIGHT;
			size_t DEVICE_IMAGE2D_MAX_WIDTH;
			size_t DEVICE_IMAGE3D_MAX_DEPTH;
			size_t DEVICE_IMAGE3D_MAX_HEIGHT;
			size_t DEVICE_IMAGE3D_MAX_WIDTH;
			cl_ulong DEVICE_LOCAL_MEM_SIZE;
			cl_device_local_mem_type DEVICE_LOCAL_MEM_TYPE;
			cl_uint DEVICE_MAX_CLOCK_FREQUENCY;
			cl_uint DEVICE_MAX_COMPUTE_UNITS;
			cl_uint DEVICE_MAX_CONSTANT_ARGS;
			cl_ulong DEVICE_MAX_CONSTANT_BUFFER_SIZE;
			cl_ulong DEVICE_MAX_MEM_ALLOC_SIZE;
			size_t DEVICE_MAX_PARAMETER_SIZE;
			cl_uint DEVICE_MAX_READ_IMAGE_ARGS;
			cl_uint DEVICE_MAX_SAMPLERS;
			size_t DEVICE_MAX_WORK_GROUP_SIZE;
			cl_uint DEVICE_MAX_WORK_ITEM_DIMENSIONS;
			size_t DEVICE_MAX_WORK_ITEM_SIZES[3];
			cl_uint DEVICE_MAX_WRITE_IMAGE_ARGS;
			cl_uint DEVICE_MEM_BASE_ADDR_ALIGN;
			cl_uint DEVICE_MIN_DATA_TYPE_ALIGN_SIZE;
			char* DEVICE_NAME;
			cl_platform_id DEVICE_PLATFORM;
			cl_uint DEVICE_PREFERRED_VECTOR_WIDTH_CHAR;
			cl_uint DEVICE_PREFERRED_VECTOR_WIDTH_SHORT;
			cl_uint DEVICE_PREFERRED_VECTOR_WIDTH_INT;
			cl_uint DEVICE_PREFERRED_VECTOR_WIDTH_LONG;
			cl_uint DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT;
			cl_uint DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE;
			char* DEVICE_PROFILE;
			size_t DEVICE_PROFILING_TIMER_RESOLUTION;
			cl_command_queue_properties DEVICE_QUEUE_PROPERTIES;
			cl_device_fp_config DEVICE_SINGLE_FP_CONFIG;
			cl_device_type DEVICE_TYPE;
			char* DEVICE_VENDOR;
			cl_uint DEVICE_VENDOR_ID;
			char* DEVICE_VERSION;
			char* DRIVER_VERSION;
		};
		
		/*!
		 *  \ingroup helpers
		 */
		/*!
		 *  \class Device_CL
		 *
		 *  \brief A class representing an OpenCL device.
		 *
		 *  This class represents one OpenCL device which can be used to execute the skeletons on if OpenCL
		 *  is used as backend. Stores various properties about the device and provides functions that return them.
		 *  Also contains a OpenCL context and queue.
		 */
		class Device_CL
		{
		private:
			
			cl_device_type m_type;
			openclDeviceProp m_deviceProp;
			
			cl_device_id m_device;
			cl_context m_context;
			cl_command_queue m_queue;
			
			size_t m_maxThreads;
			size_t m_maxBlocks;
			
		public:
			/*!
			 *  The constructor creates a device from an ID, device type (should be GPU in this version)
			 *  and a context. It gets all the properties and creates a command-queue.
			 *
			 *  \param id Device ID for the device that is to be created.
			 *  \param type The OpenCL device type.
			 *  \param context A valid OpenCL context.
			 */
			Device_CL(cl_device_id id, cl_device_type type, cl_context context)
			: m_device(id), m_type(type), m_context(context)
			{
				for (openclGenProp &prop : this->m_deviceProp.propertyList)
				{
					cl_int err = clGetDeviceInfo(this->m_device, prop.param_name, prop.param_value_size, prop.param_value, NULL);
					CL_CHECK_ERROR(err, "Error adding OpenCL property value");
				}
				
				this->m_maxThreads = this->getMaxBlockSize() >> 1;
				this->m_maxBlocks = (size_t)((size_t)1 << (m_deviceProp.DEVICE_ADDRESS_BITS-1)) * 2 - 1;
				
				// Create a command-queue on the GPU device
				cl_int err;
				m_queue = clCreateCommandQueue(m_context, m_device, 0, &err);
				CL_CHECK_ERROR(err, "Error creating OpenCL command queue");
			}
			
			//! The destructor releases the OpenCL queue and context.
			~Device_CL()
			{
				clReleaseCommandQueue(m_queue);
				clReleaseContext(m_context);
				DEBUG_TEXT_LEVEL1("Released Device_CL");
			}
			
			//! \return The maximum block (work group) size.
			size_t getMaxBlockSize() const
			{
				return m_deviceProp.DEVICE_MAX_WORK_GROUP_SIZE;
			}
			
			//! \return The maximum number of compute units available.
			cl_uint getNumComputeUnits() const
			{
				return m_deviceProp.DEVICE_MAX_COMPUTE_UNITS;
			}
			
			//!  \return The global memory size.
			cl_ulong getGlobalMemSize() const
			{
				return m_deviceProp.DEVICE_GLOBAL_MEM_SIZE;
			}
			
			//! \return The local (shared) memory size.
			cl_ulong getSharedMemPerBlock() const
			{
				return m_deviceProp.DEVICE_LOCAL_MEM_SIZE;
			}
			
			///! \return The maximum number of threads per block or group.
			int getMaxThreads() const
			{
#ifdef SKEPU_MAX_GPU_THREADS
				return SKEPU_MAX_GPU_THREADS;
#else
				return m_maxThreads;
#endif
			}
			
			//! \return The maximum number of blocks or groups for a kernel launch.
			size_t getMaxBlocks() const
			{
#ifdef SKEPU_MAX_GPU_BLOCKS
				return SKEPU_MAX_GPU_BLOCKS;
#else
				return m_maxBlocks;
#endif
			}
			
			//! \return OpenCL context.
			const cl_context& getContext() const
			{
				return m_context;
			}
			
			//! \return OpenCL command queue.
			const cl_command_queue& getQueue() const
			{
				return m_queue;
			}
			
			//! \return OpenCL type.
			cl_device_type getType() const
			{
				return m_type;
			}
			
			//! \return OpenCL device id.
			cl_device_id getDeviceID() const
			{
				return m_device;
			}
			
		};
		
	} // end namespace backend
} // end namespace skepu2

#endif // SKEPU_OPENCL

#endif // DEVICE_CL_H
