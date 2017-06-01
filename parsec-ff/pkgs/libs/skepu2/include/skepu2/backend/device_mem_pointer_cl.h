/*! \file device_mem_pointer_cl.h
 *  \brief Contains a class declaration for an object which represents an OpenCL device memory allocation for container.
 */

#ifndef DEVICE_MEM_POINTER_CL_H
#define DEVICE_MEM_POINTER_CL_H

#ifdef SKEPU_OPENCL

#include "environment.h"
#include "device_cl.h"

namespace skepu2
{
	namespace backend
	{
		/*!
		 *  \ingroup helpers
		 */
		/*!
		 *  \class DeviceMemPointer_CL
		 *
		 *  \brief A class representing an OpenCL device memory allocation for container.
		 *
		 *  This class represents an OpenCL device 1D memory allocation and controls the data transfers between
		 *  host and device.
		 */
		template <typename T>
		class DeviceMemPointer_CL
		{
		public:
			DeviceMemPointer_CL(T* root, T* start, size_t numElements, Device_CL* device);
			DeviceMemPointer_CL(T* start, size_t numElements, Device_CL* device);
			~DeviceMemPointer_CL();
			
			operator DeviceMemPointer_CL<const T>()
			{
				return *this;
			}
			
			void copyHostToDevice(size_t numElements = -1, bool copyLast=false) const;
			void copyDeviceToHost(size_t numElements = -1, bool copyLast=false) const;
			void copyDeviceToDevice(cl_mem copyToPointer, size_t numElements, size_t dstOffset = 0, size_t srcOffset = 0) const;
			
			cl_mem getDeviceDataPointer() const;
			void changeDeviceData(bool condition = true);
			
			// marks first initialization, useful when want to separate actual OpenCL allocation and memory copy (HTD) such as when using mulit-GPU OpenCL.
			mutable bool m_initialized = false;
			
		private:
			void copyHostToDevice_internal(T* src, cl_mem dest, size_t numElements, size_t offset = 0) const;
			
			T* m_rootHostDataPointer;
			T* m_effectiveHostDataPointer;
			T* m_hostDataPointer;
			
			size_t m_effectiveNumElements;
			
			cl_mem m_deviceDataPointer;
			cl_mem m_effectiveDeviceDataPointer;
			
			size_t m_numElements;
			Device_CL* m_device;
			
			mutable bool m_deviceDataHasChanged = false;
		};
		
		
		/*!
		 *  The constructor allocates a certain amount of space in device memory and stores a pointer to
		 *  some data in host memory. Takes a root address as well.
		 *
		 *  \param root Pointer to starting address of data in host memory (can be same as start).
		 *  \param start Pointer to data in host memory.
		 *  \param numElements Number of elements to allocate memory for.
		 *  \param device Pointer to a valid device to allocate the space on.
		 */
		template <typename T>
		DeviceMemPointer_CL<T>::DeviceMemPointer_CL(T* root, T* start, size_t numElements, Device_CL* device) : m_effectiveHostDataPointer(start), m_hostDataPointer(root), m_numElements(numElements), m_effectiveNumElements(numElements), m_device(device)
		{
#ifdef SKEPU_MEASURE_TIME_DISTRIBUTION
#ifdef SKEPU_MEASURE_ONLY_COPY
			clFinish(m_device->getQueue());
#endif
			devMemAllocTimer.start();
#endif
			
			cl_int err;
			size_t sizeVec = numElements * sizeof(T);
			m_deviceDataPointer = clCreateBuffer(m_device->getContext(), CL_MEM_READ_WRITE, sizeVec, NULL, &err);
			CL_CHECK_ERROR(err, "Error allocating memory on OpenCL device, size: " << sizeVec);
			
			DEBUG_TEXT_LEVEL1("Alloc OpenCL, ptr: " << m_deviceDataPointer << ", size: " << sizeVec);
			
			m_effectiveDeviceDataPointer = m_deviceDataPointer;
			
#ifdef SKEPU_MEASURE_TIME_DISTRIBUTION
#ifdef SKEPU_MEASURE_ONLY_COPY
			clFinish(m_device->getQueue());
#endif
			devMemAllocTimer.stop();
#endif
		}
		
		
		
		/*!
		 *  The constructor allocates a certain amount of space in device memory and stores a pointer to
		 *  some data in host memory.
		 *
		 *  \param start Pointer to data in host memory.
		 *  \param numElements Number of elements to allocate memory for.
		 *  \param device Pointer to a valid device to allocate the space on.
		 */
		template <typename T>
		DeviceMemPointer_CL<T>::DeviceMemPointer_CL(T* start, size_t numElements, Device_CL* device)
		: DeviceMemPointer_CL<T>::DeviceMemPointer_CL(start, start, numElements, device)
		{}
		
		
		//! The destructor releases the allocated device memory.
		template <typename T>
		DeviceMemPointer_CL<T>::~DeviceMemPointer_CL()
		{
			DEBUG_TEXT_LEVEL1("Dealloc OpenCL, ptr: " << m_deviceDataPointer << ", size: " << m_numElements);
			clReleaseMemObject(m_deviceDataPointer);
		}
		
		
		/*!
		 *  Copies data from device memory to another device memory.
		 *
		 *  \param copyToPointer The destination address.
		 *  \param numElements Number of elements to copy, default value -1 = all elements.
		 *  \param dstOffset Offset (if any) in destination pointer.
		 *  \param srcOffset Offset (if any) in source pointer.
		 */
		template <typename T>
		void DeviceMemPointer_CL<T>::copyDeviceToDevice(cl_mem copyToPointer, size_t numElements, size_t dstOffset, size_t srcOffset) const
		{
			if(m_hostDataPointer != NULL)
			{
#ifdef SKEPU_MEASURE_TIME_DISTRIBUTION
#ifdef SKEPU_MEASURE_ONLY_COPY
				clFinish(m_device->getQueue());
#endif
				copyUpTimer.start();
#endif
				
				size_t sizeVec = ((numElements != -1) ? numElements : m_numElements) * sizeof(T);
				cl_int err = clEnqueueCopyBuffer(m_device->getQueue(),m_deviceDataPointer, copyToPointer, srcOffset*sizeof(T), dstOffset*sizeof(T), sizeVec, 0, NULL, NULL);
				CL_CHECK_ERROR(err, "Error copying data to OpenCL device, size: " << sizeVec);
				
				DEBUG_TEXT_LEVEL1("DEVICE_TO_DEVICE OpenCL, size " << sizeVec);
				
#ifdef SKEPU_MEASURE_TIME_DISTRIBUTION
#ifdef SKEPU_MEASURE_ONLY_COPY
				clFinish(m_device->getQueue());
#endif
				copyUpTimer.stop();
#endif
				m_deviceDataHasChanged = true;
			}
		}
		
		
		/*!
		*  Copies data from host memory to device memory. An internal method not used publcially.
		*  It allows copying with offset
		*
		*  \param src_ptr The source address.
		*  \param dst_ptr The destination address.
		*  \param numElements Number of elements to copy, default value -1 = all elements.
		*  \param offset The offset in the device buffer.
 */
template <typename T>
		void DeviceMemPointer_CL<T>::copyHostToDevice_internal(T* src_ptr, cl_mem dest_ptr, size_t numElements, size_t offset) const
		{
#ifdef SKEPU_MEASURE_TIME_DISTRIBUTION
#ifdef SKEPU_MEASURE_ONLY_COPY
			clFinish(m_device->getQueue());
#endif
			copyUpTimer.start();
#endif
			
			size_t sizeVec = numElements*sizeof(T);
			cl_int err = clEnqueueWriteBuffer(m_device->getQueue(), dest_ptr, CL_TRUE, offset*sizeof(T), sizeVec, (void*)src_ptr, 0, NULL, NULL);
			CL_CHECK_ERROR(err, "Error copying data to OpenCL device, size: " << sizeVec);
			
			DEBUG_TEXT_LEVEL1("HOST_TO_DEVICE INTERNAL OpenCL, size " << sizeVec);
			
#ifdef SKEPU_MEASURE_TIME_DISTRIBUTION
#ifdef SKEPU_MEASURE_ONLY_COPY
			clFinish(m_device->getQueue());
#endif
			copyUpTimer.stop();
#endif
			
		m_initialized = true;
	//	m_deviceDataHasChanged = false;
		}
		
		
		/*!
		 *  Copies data from host memory to device memory.
		 *
		 *  \param numElements Number of elements to copy, default value -1 = all elements.
		 *  \param copyLast Boolean flag specifying whether should copy last updated copy only (default: false).
		 */
		template <typename T>
		void DeviceMemPointer_CL<T>::copyHostToDevice(size_t numElements, bool copyLast) const
		{
			cl_int err;
			size_t sizeVec;
			if (numElements == -1)
				sizeVec = (copyLast ? m_numElements : m_effectiveNumElements) * sizeof(T);
			else
				sizeVec = numElements*sizeof(T);
			
#ifdef SKEPU_MEASURE_TIME_DISTRIBUTION
#ifdef SKEPU_MEASURE_ONLY_COPY
			clFinish(m_device->getQueue());
#endif
			copyUpTimer.start();
#endif
			
			if (copyLast)
				err = clEnqueueWriteBuffer(m_device->getQueue(), m_deviceDataPointer, CL_TRUE, 0, sizeVec, m_hostDataPointer, 0, NULL, NULL);
			else
				err = clEnqueueWriteBuffer(m_device->getQueue(), m_effectiveDeviceDataPointer, CL_TRUE, 0, sizeVec, m_effectiveHostDataPointer, 0, NULL, NULL);
			CL_CHECK_ERROR(err, "Error copying data to OpenCL device, size: " << sizeVec);
			
			DEBUG_TEXT_LEVEL1("HOST_TO_DEVICE OpenCL, size " << sizeVec);
			
#ifdef SKEPU_MEASURE_TIME_DISTRIBUTION
#ifdef SKEPU_MEASURE_ONLY_COPY
			clFinish(m_device->getQueue());
#endif
			copyUpTimer.stop();
#endif
			
			m_initialized = true;
			m_deviceDataHasChanged = false;
		}
		
		
		/*!
		 *  Copies data from device memory to host memory. Only copies if data on device has been marked as changed.
		 *
		 *  \param numElements Number of elements to copy, default value -1 = all elements.
		 *  \param copyLast Boolean flag specifying whether should copy last updated copy only (default: false).
		 */
		template <typename T>
		void DeviceMemPointer_CL<T>::copyDeviceToHost(size_t numElements, bool copyLast) const
		{
			if (m_deviceDataHasChanged)
			{
				cl_int err;
				size_t sizeVec;
				if (numElements == -1)
					sizeVec = (copyLast ? m_numElements : m_effectiveNumElements) * sizeof(T);
				else
					sizeVec = numElements*sizeof(T);
				
#ifdef SKEPU_MEASURE_TIME_DISTRIBUTION
#ifdef SKEPU_MEASURE_ONLY_COPY
				clFinish(m_device->getQueue());
#endif
				copyDownTimer.start();
#endif
				
				if(copyLast)
					err = clEnqueueReadBuffer(m_device->getQueue(), m_deviceDataPointer, CL_TRUE, 0, sizeVec, (void*)m_hostDataPointer, 0, NULL, NULL);
				else
					err = clEnqueueReadBuffer(m_device->getQueue(), m_effectiveDeviceDataPointer, CL_TRUE, 0, sizeVec, (void*)m_effectiveHostDataPointer, 0, NULL, NULL);
				CL_CHECK_ERROR(err, "Error copying data from OpenCL device, size: " << sizeVec);
				
				DEBUG_TEXT_LEVEL1("DEVICE_TO_HOST OpenCL, size " << sizeVec);
				
#ifdef SKEPU_MEASURE_TIME_DISTRIBUTION
#ifdef SKEPU_MEASURE_ONLY_COPY
				clFinish(m_device->getQueue());
#endif
				copyDownTimer.stop();
#endif
				m_deviceDataHasChanged = false;
			}
		}
		
		/// \return OpenCL memory object representing data on the device.
		template <typename T>
		cl_mem DeviceMemPointer_CL<T>::getDeviceDataPointer() const
		{
		//	std::cout << m_deviceDataPointer << std::endl;
			return m_deviceDataPointer;
		}
		
		//! Marks the device data as changed.
		template <typename T>
		void DeviceMemPointer_CL<T>::changeDeviceData(bool condition)
		{
			if (condition)
			{
				DEBUG_TEXT_LEVEL1("CHANGE_DEVICE_DATA OpenCL");
				m_deviceDataHasChanged = true;
				m_initialized = true;
			}
		}
		
		template<typename T>
		DeviceMemPointer_CL<const T> *device_mem_pointer_const_cast(DeviceMemPointer_CL<T> *ptr)
		{
			return (DeviceMemPointer_CL<const T> *)(ptr);
		}
		
	} // end namespace backend
} // end namespace skepu2

#endif // SKEPU_OPENCL

#endif // DEVICE_MEM_POINTER_CL_H
