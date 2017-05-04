/*! \file matrix_cl.inl
 *  \brief Contains the definitions of OpenCL specific member functions of the Matrix class.
 */

#ifdef SKEPU_OPENCL

namespace skepu2
{
	/*!
	 *  \brief Update device with matrix content.
	 *
	 *  Update device with a Matrix range by specifying rowsize and column size. This allows to create rowwise paritions.
	 *  If Matrix does not have an allocation on the device for
	 *  the current range, create a new allocation and if specified, also copy Matrix data to device.
	 *  Saves newly allocated ranges to \p m_deviceMemPointers_CL so matrix can keep track of where
	 *  and what it has stored on devices.
	 *
	 *  \param start Pointer to first element in range to be updated with device.
	 *  \param rows Number of rows.
	 *  \param cols Number of columns.
	 *  \param device Pointer to the device that should be synched with.
	 *  \param copy Boolean value that tells whether to only allocate or also copy matrix data to device. True copies, False only allocates.
	 */
	template <typename T>
	typename Matrix<T>::device_pointer_type_cl Matrix<T>::updateDevice_CL(T* start, size_type rows, size_type cols, backend::Device_CL* device, bool copy)
	{
		DEBUG_TEXT_LEVEL3("Matrix updating device OpenCL\n")
		
		std::pair<cl_device_id, std::pair<const T*, size_type>> key(device->getDeviceID(), std::pair<const T*, size_type>(start, rows * cols));
		auto result = m_deviceMemPointers_CL.find(key);
		
		if (result == m_deviceMemPointers_CL.end()) //insert new, alloc mem and copy
		{
			auto temp = new backend::DeviceMemPointer_CL<T>(start, rows * cols, device);
			if(copy)
			{
				// Make sure uptodate
				updateHost_CL();
				// Copy
				temp->copyHostToDevice();
			}
			result = m_deviceMemPointers_CL.insert(m_deviceMemPointers_CL.begin(), std::make_pair(key,temp));
		}
		else if (copy && !result->second->m_initialized) // we check for case when space is allocated but data was not copied, Multi-GPU case
		{
			// Make sure uptodate
			updateHost_CL(); // FIX IT: Only check for this copy and not for all copies.
			// Copy
			result->second->copyHostToDevice();	 // internally it will set "result->second->m_initialized = true; "
		}
		return result->second;
	}
	
	
	template <typename T>
	typename Matrix<T>::device_const_pointer_type_cl Matrix<T>::updateDevice_CL(const T* start, size_type rows, size_type cols, backend::Device_CL* device, bool copy) const
	{
		DEBUG_TEXT_LEVEL3("Matrix updating device OpenCL\n")
			
		std::pair<cl_device_id, std::pair<const T*, size_type>> key(device->getDeviceID(), std::pair<const T*, size_type>(start, rows * cols));
		auto result = m_deviceMemPointers_CL.find(key);
		
		if (result == this->m_deviceConstMemPointers_CL.end()) //insert new, alloc mem and copy
		{
			auto temp = new device_const_pointer_type_cl(start, rows * cols, device);
			if (copy)
				temp->copyHostToDevice();
			
			result = this->m_deviceConstMemPointers_CL.insert(m_deviceMemPointers_CL.begin(), std::make_pair(key, temp));
		}
		else if (copy && !result->second->m_initialized) // we check for case when space is allocated but data was not copied, Multi-GPU case
		{
			result->second->copyHostToDevice();	 // internally it will set "result->second->m_initialized = true; "
		}
		return result->second;
	}
	
	
	/*!
	 *  \brief Update device with matrix content.
	 *
	 *  Update device with a Matrix range by specifying rowsize only as number of rows is assumed to be 1 in this case.
	 *  Helper function, useful for scenarios where matrix need to be treated like Vector 1D.
	 *  If Matrix does not have an allocation on the device for
	 *  the current range, create a new allocation and if specified, also copy Matrix data to device.
	 *  Saves newly allocated ranges to \p m_deviceMemPointers_CL so matrix can keep track of where
	 *  and what it has stored on devices.
	 *
	 *  \param start Pointer to first element in range to be updated with device.
	 *  \param cols Number of columns.
	 *  \param device Pointer to the device that should be synched with.
	 *  \param copy Boolean value that tells whether to only allocate or also copy matrix data to device. True copies, False only allocates.
	 */
	template <typename T>
	typename Matrix<T>::device_pointer_type_cl Matrix<T>::updateDevice_CL(T* start, size_type cols, backend::Device_CL* device, bool copy)
	{
		return updateDevice_CL(start, (size_type)1, cols, device, copy);
	}
	
	template <typename T>
	typename Matrix<T>::device_const_pointer_type_cl Matrix<T>::updateDevice_CL(const T* start, size_type cols, backend::Device_CL* device, bool copy) const
	{
		return updateDevice_CL(start, (size_type)1, cols, device, copy);
	}
	
	/*!
	 *  \brief Flushes the matrix.
	 *
	 *  First it updates the matrix from all its device allocations, then it releases all allocations.
	 */
	template <typename T>
	void Matrix<T>::flush_CL()
	{
		DEBUG_TEXT_LEVEL3("Matrix flush OpenCL\n")

			updateHost_CL();
		releaseDeviceAllocations_CL();
	}

	/*!
	*  \brief Updates the host from devices.
	*
	*  Updates the matrix from all its device allocations.
 */
template <typename T>
	inline void Matrix<T>::updateHost_CL() const
	{
		DEBUG_TEXT_LEVEL3("Matrix updating host OpenCL\n")

			if (!this->m_deviceMemPointers_CL.empty())
				for (auto &memptr : this->m_deviceMemPointers_CL)
					memptr.second->copyDeviceToHost();
	}

	/*!
	*  \brief Invalidates the device data.
	*
	*  Invalidates the device data by releasing all allocations. This way the matrix is updated
	*  and then data must be copied back to devices if used again.
 */
template <typename T>
	inline void Matrix<T>::invalidateDeviceData_CL() const
	{
		DEBUG_TEXT_LEVEL3("Matrix invalidating device data OpenCL\n")

   //deallocs all device mem for matrix for now
			if (!this->m_deviceMemPointers_CL.empty())
				releaseDeviceAllocations_CL();
   //Could maybe be made better by only setting a flag that data is not valid
	}

	/*!
	*  \brief Releases device allocations.
	*
	*  Releases all device allocations for this matrix. The memory pointers are removed.
 */
template <typename T>
	inline void Matrix<T>::releaseDeviceAllocations_CL() const
	{
		DEBUG_TEXT_LEVEL3("Matrix releasing device allocations OpenCL\n")

			for (auto &memptr : this->m_deviceMemPointers_CL)
				delete memptr.second;
   
		m_deviceMemPointers_CL.clear();
	}

}

#endif
