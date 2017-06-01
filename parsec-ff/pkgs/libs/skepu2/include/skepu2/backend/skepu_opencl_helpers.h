/*! \file skepu_opencl_helpers.h
 *  \brief Contains the definitions of some helper functions related to \em OpenCL backend.
 */

#ifndef SKEPU_OPENCL_HELPERS_H
#define SKEPU_OPENCL_HELPERS_H

#ifdef SKEPU_OPENCL

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "device_cl.h"


namespace skepu2
{
	template<typename T>
	class Vector;
	
	template<typename T>
	class Matrix;
	
	template<typename T>
	class SparseMatrix;
	
	namespace backend
	{
		template<typename T>
		class DeviceMemPointer_CL;
		
		namespace cl_helpers
		{

/*#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif*/


// Give a little more for Windows : the console window often disapears before we can read the message
#ifdef _WIN32
# if 1//ndef UNICODE
#  ifdef _DEBUG // Do this only in debug mode...
			inline void VSPrintf(FILE *file, LPCSTR fmt, ...)
			{
				size_t fmt2_sz	= 2048;
				char *fmt2		= (char*)malloc(fmt2_sz);
				va_list  vlist;
				va_start(vlist, fmt);
				while((_vsnprintf(fmt2, fmt2_sz, fmt, vlist)) < 0) // means there wasn't anough room
				{
					fmt2_sz *= 2;
					if(fmt2) free(fmt2);
					fmt2 = (char*)malloc(fmt2_sz);
				}
				OutputDebugStringA(fmt2);
				fprintf(file, fmt2);
				free(fmt2);
			}
#	define FPRINTF(a) VSPrintf a
#  else //debug
#	define FPRINTF(a) fprintf a
// For other than Win32
#  endif //debug
# else //unicode
// Unicode case... let's give-up for now and keep basic printf
#	define FPRINTF(a) fprintf a
# endif //unicode
#else //win32
#	define FPRINTF(a) fprintf a
#endif //win32

			template<size_t... I, typename... Args>
			inline void setKernelArgsHelper(cl_kernel kernel, pack_indices<I...>, Args... args)
			{
				cl_int errors[sizeof...(Args)] = 
				{
					clSetKernelArg(kernel, I, sizeof get<I, Args...>(args...), &get<I, Args...>(args...))...,
				};
				
				for (size_t e = 0; e < sizeof...(Args); ++e)
					CL_CHECK_ERROR(errors[e], "Error in OpenCL kernel argument nr " << e);
			}
			
			template<typename... Args>
			inline void setKernelArgs(cl_kernel kernel, Args... args)
			{
				setKernelArgsHelper(kernel, typename make_pack_indices<sizeof...(Args), 0>::type(), args...);
			}
			
			template<typename T>
			inline std::tuple<Vector<T> *, DeviceMemPointer_CL<T> *>
			randomAccessArg(Vector<T> &c, Device_CL *device, bool read)
			{
				auto memP = c.updateDevice_CL(c.getAddress(), c.size(), device, read);
				
				return std::make_tuple(&c, memP);
			}
			
			template<typename T>
			inline std::tuple<Matrix<T> *, DeviceMemPointer_CL<T> *>
			randomAccessArg(Matrix<T> &c, Device_CL *device, bool read)
			{
				auto memP = c.updateDevice_CL(c.getAddress(), c.size(), device, read);
				
				return std::make_tuple(&c, memP);
			}
			
			template<typename T>
			inline std::tuple<SparseMatrix<T> *, DeviceMemPointer_CL<T> *, DeviceMemPointer_CL<size_t> *, DeviceMemPointer_CL<size_t> *>
			randomAccessArg(SparseMatrix<T> &c, Device_CL *device, bool /* ignored, always read-only */)
			{
				auto valuesMemP     = c.updateDevice_CL(      c.get_values(),       c.total_nnz(), device, true);
				auto rowOffsetsMemP = c.updateDevice_Index_CL(c.get_row_pointers(), c.total_rows() + 1, device, true);
				auto colIndicesMemP = c.updateDevice_Index_CL(c.get_col_indices(),  c.total_nnz(), device, true);
				
				return std::make_tuple(&c, valuesMemP, rowOffsetsMemP, colIndicesMemP);
			}
			
			template <typename T>
			inline void copyDeviceToHost(T *hostPtr, cl_mem devPtr, size_t numElements, Device_CL* device, size_t offset)
			{
				if (devPtr != NULL && hostPtr != NULL)
				{
					DEBUG_TEXT_LEVEL2("** DEVICE_TO_HOST OpenCL: "<< numElements)
					
					cl_int err = clEnqueueReadBuffer(device->getQueue(), devPtr, CL_TRUE, offset, numElements*sizeof(T), hostPtr, 0, NULL, NULL);
					CL_CHECK_ERROR(err, "Error copying data from OpenCL device");
				}
			}
			
			
			template <typename T>
			inline void copyHostToDevice(T *hostPtr, cl_mem devPtr, size_t numElements, Device_CL* device, size_t offset)
			{
				if (hostPtr != NULL && devPtr != NULL)
				{
					DEBUG_TEXT_LEVEL2("** HOST_TO_DEVICE OpenCL: " << numElements)
					
					cl_int err = clEnqueueWriteBuffer(device->getQueue(), devPtr, CL_TRUE, offset, numElements * sizeof(T), hostPtr, 0, NULL, NULL);
					CL_CHECK_ERROR(err, "Error copying data to OpenCL device");
				}
			}
			
			
			template<typename T>
			inline cl_buffer_region makeBufferRegion(size_t origin, size_t size)
			{
				cl_buffer_region info;
				info.origin = origin * sizeof(T);
				info.size = size * sizeof(T);
				return info;
			}
			
			
			template <typename T>
			inline cl_mem allocateDeviceMemory(size_t size, Device_CL* device)
			{
				DEBUG_TEXT_LEVEL2("** ALLOC OpenCL: " << size)
				
				cl_int err;
				cl_mem devicePointer = clCreateBuffer(device->getContext(), CL_MEM_READ_WRITE, size * sizeof(T), NULL, &err);
				CL_CHECK_ERROR(err, "Error allocating memory on OpenCL device");
				return devicePointer;
			}
			
			
			template <typename T>
			inline void freeDeviceMemory(cl_mem d_pointer)
			{
				DEBUG_TEXT_LEVEL2("** DE-ALLOC OpenCL")
				
				//	if(d_pointer!=NULL)
				{
					cl_int err = clReleaseMemObject(d_pointer);
					CL_CHECK_ERROR(err, "Error releasing memory on device");
				}
			}
			
			
			/*!
			 *  A helper function used by createOpenCLProgram(). It finds all instances of a string in another string and replaces it with
			 *  a third string.
			 *
			 *  \param text A \p std::string which is searched.
			 *  \param find The \p std::string which is searched for and replaced.
			 *  \param replace The relpacement \p std::string.
			 */
			inline void replaceTextInString(std::string& text, std::string find, std::string replace)
			{
				std::string::size_type pos=0;
				while((pos = text.find(find, pos)) != std::string::npos)
				{
					text.erase(pos, find.length());
					text.insert(pos, replace);
					pos+=replace.length();
				}
			}
			
			
			inline std::string replaceSizeT(std::string const& source)
			{
				std::string replacedSource = source;
				
				// Replace usage of size_t to match host platform size
				if (sizeof(size_t) <= sizeof(unsigned int))
					replaceTextInString(replacedSource, std::string("size_t "), "unsigned int ");
				else if (sizeof(size_t) <= sizeof(unsigned long))
					replaceTextInString(replacedSource, std::string("size_t "), "unsigned long ");
				else
					SKEPU_ERROR("OpenCL code compilation issue: sizeof(size_t) is bigger than 8 bytes: " << sizeof(size_t));
				
				return replacedSource;
			}
			
			
			/*!
			 * A helper function for OpenCL backends. It takes an OpenCL error code and prints the corresponding error message
			 *
			 * \param Err OpenCL error
			 * \param s Optional text string that may give more information on the error source
			 */
			inline void printCLError(cl_int Err, std::string s)
			{
				std::string msg;
				if (Err != CL_SUCCESS)
				{
					switch(Err)
					{
					case CL_DEVICE_NOT_FOUND:
						msg = "Device not found"; break;
					case CL_DEVICE_NOT_AVAILABLE:
						msg = "Device not available"; break;
					case CL_COMPILER_NOT_AVAILABLE:
						msg = "Compiler not available"; break;
					case CL_MEM_OBJECT_ALLOCATION_FAILURE:
						msg = "Memory object allocation failure"; break;
					case CL_OUT_OF_RESOURCES:
						msg = "Out of resources"; break;
					case CL_OUT_OF_HOST_MEMORY:
						msg = "Out of host memory"; break;
					case CL_PROFILING_INFO_NOT_AVAILABLE:
						msg = "Profiling info not available"; break;
					case CL_MEM_COPY_OVERLAP:
						msg = "Memory copy overlap"; break;
					case CL_IMAGE_FORMAT_MISMATCH:
						msg = "Image format mismatch"; break;
					case CL_IMAGE_FORMAT_NOT_SUPPORTED:
						msg = "Image format not supported"; break;
					case CL_BUILD_PROGRAM_FAILURE:
						msg = "Build program failure"; break;
					case CL_MAP_FAILURE:
						msg = "Map failure"; break;
					case CL_MISALIGNED_SUB_BUFFER_OFFSET:
						msg = "Misaligned sub buffer offset"; break;
					case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
						msg = "Exec status error for events in wait list"; break;
					case CL_INVALID_VALUE:
						msg = "Invalid value"; break;
					case CL_INVALID_DEVICE_TYPE:
						msg = "Invalid device type"; break;
					case CL_INVALID_PLATFORM:
						msg = "Invalid platform"; break;
					case CL_INVALID_DEVICE:
						msg = "Invalid device"; break;
					case CL_INVALID_CONTEXT:
						msg = "Invalid context"; break;
					case CL_INVALID_QUEUE_PROPERTIES:
						msg = "Invalid queue properties"; break;
					case CL_INVALID_COMMAND_QUEUE:
						msg = "Invalid command queue"; break;
					case CL_INVALID_HOST_PTR:
						msg = "Invalid host pointer"; break;
					case CL_INVALID_MEM_OBJECT:
						msg = "Invalid memory object"; break;
					case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
						msg = "Invalid image format descriptor"; break;
					case CL_INVALID_IMAGE_SIZE:
						msg = "Invalid image size"; break;
					case CL_INVALID_SAMPLER:
						msg = "Invalid sampler"; break;
					case CL_INVALID_BINARY:
						msg = "Invalid binary"; break;
					case CL_INVALID_BUILD_OPTIONS:
						msg = "Invalid build options"; break;
					case CL_INVALID_PROGRAM:
						msg = "Invalid program"; break;
					case CL_INVALID_PROGRAM_EXECUTABLE:
						msg = "Invalid program executable"; break;
					case CL_INVALID_KERNEL_NAME:
						msg = "Invalid kernel name"; break;
					case CL_INVALID_KERNEL_DEFINITION:
						msg = "Invalid kernel definition"; break;
					case CL_INVALID_KERNEL:
						msg = "Invalid kernel"; break;
					case CL_INVALID_ARG_INDEX:
						msg = "Invalid argument index"; break;
					case CL_INVALID_ARG_VALUE:
						msg = "Invalid argument value"; break;
					case CL_INVALID_ARG_SIZE:
						msg = "Invalid argument size"; break;
					case CL_INVALID_KERNEL_ARGS:
						msg = "Invalid kernel arguments"; break;
					case CL_INVALID_WORK_DIMENSION:
						msg = "Invalid work dimension"; break;
					case CL_INVALID_WORK_GROUP_SIZE:
						msg = "Invalid work group size"; break;
					case CL_INVALID_WORK_ITEM_SIZE:
						msg = "Invalid work item size"; break;
					case CL_INVALID_GLOBAL_OFFSET:
						msg = "Invalid global offset"; break;
					case CL_INVALID_EVENT_WAIT_LIST:
						msg = "Invalid event wait list"; break;
					case CL_INVALID_EVENT:
						msg = "Invalid event"; break;
					case CL_INVALID_OPERATION:
						msg = "Invalid operation"; break;
					case CL_INVALID_GL_OBJECT:
						msg = "Invalid GL object"; break;
					case CL_INVALID_BUFFER_SIZE:
						msg = "Invalid buffer size"; break;
					case CL_INVALID_MIP_LEVEL:
						msg = "Invalid MIP level"; break;
					case CL_INVALID_GLOBAL_WORK_SIZE:
						msg = "Invalid global work size"; break;
					case CL_INVALID_PROPERTY:
						msg = "Invalid property"; break;
					default:
						msg = "Unknown error"; break;
					}
					SKEPU_ERROR(s << " OpenCL error code " << Err << " " << msg);
				}
			}
			
			
			inline cl_program buildProgram(Device_CL *device, const std::string &source, const std::string &options = "", size_t deviceID = 0)
			{
				cl_int err;
				std::stringstream allOptionsStream;
				allOptionsStream << options << " -DSKEPU_INTERNAL_DEVICE_ID=" << deviceID;
				const std::string allOptions = allOptionsStream.str();
				
				const char *src = source.c_str();
				const char *opt = allOptions.c_str();
				cl_program program = clCreateProgramWithSource(device->getContext(), 1, &src, NULL, &err);
				CL_CHECK_ERROR(err, "Error creating OpenCL program");    
				
				err = clBuildProgram(program, 0, NULL, opt, NULL, NULL);
				if (err != CL_SUCCESS)
				{
#if true
					cl_build_status build_status;
					clGetProgramBuildInfo(program, device->getDeviceID(), CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);
					if (build_status != CL_SUCCESS)
					{
						size_t log_size;
						clGetProgramBuildInfo(program, device->getDeviceID(), CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
						std::string buildLog(log_size, '\0');
						clGetProgramBuildInfo(program, device->getDeviceID(), CL_PROGRAM_BUILD_LOG, log_size, &buildLog[0], NULL);
						SKEPU_ERROR("Error building OpenCL program: " << err << "\n" << buildLog);
					}
#else
					SKEPU_ERROR("Error building OpenCL program: " << err);
#endif
				}
				return program;
			}
			
		} // end namespace cl_helpers
		
	} // end namespace backend
	
} // end namespace skepu2

#endif // SKEPU_OPENCL

#endif // SKEPU_OPENCL_HELPERS_H
