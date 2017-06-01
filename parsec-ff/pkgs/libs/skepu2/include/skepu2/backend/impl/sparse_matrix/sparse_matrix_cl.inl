/*! \file sparse_matrix_cl.inl
 *  \brief Contains the definitions of member functions of the SparseMatrix class related to \em OpenCL backend.
 */

#ifdef SKEPU_OPENCL

namespace skepu2
{

/*!
 *  \brief Update device with sparse matrix content.
 *
 *  Update device with a SparseMatrix range by specifying number of elements.
 *  If sparse matrix does not have an allocation on the device for
 *  the current range, create a new allocation and if specified, also copy sparse matrix data to device.
 *  Saves newly allocated ranges to \p m_deviceMemPointers_CL so sparse matrix can keep track of where
 *  and what it has stored on devices.
 *
 *  \param start Pointer to first element in range to be updated with device.
 *  \param elems Number of (non-zero) elements.
 *  \param device Pointer to the device that should be synched with.
 *  \param copy Boolean value that tells whether to only allocate or also copy sparse matrix data to device. True copies, False only allocates.
 */
template <typename T>
typename SparseMatrix<T>::device_pointer_type_cl SparseMatrix<T>::updateDevice_CL(T* start, size_t elems, backend::Device_CL* device, bool copy)
{
   DEBUG_TEXT_LEVEL3("SparseMatrix updating device CUDA\n")

   typename std::map<std::pair< cl_device_id, std::pair< T*, size_t > >, device_pointer_type_cl >::iterator result;

   std::pair< cl_device_id, std::pair< T*, size_t > > key(device->getDeviceID(), std::pair< T*, size_t >(start, elems));
   result = m_deviceMemPointers_CL.find(key);

   if(result == m_deviceMemPointers_CL.end()) //insert new, alloc mem and copy
   {
      device_pointer_type_cl temp = new backend::DeviceMemPointer_CL<T>(start, elems, device);
      if(copy)
      {
         //Make sure uptodate
         updateHost_CL();
         //Copy
         temp->copyHostToDevice();
      }
      result = m_deviceMemPointers_CL.insert(m_deviceMemPointers_CL.begin(), std::make_pair(key,temp));
   }
   else if(copy && !result->second->m_initialized) // we check for case when space is allocated but data was not copied, Multi-GPU case
   {
      //Make sure uptodate
      updateHost_CL(); // FIX IT: Only check for this copy and not for all copies.
      //Copy
      result->second->copyHostToDevice();	 // internally it will set "result->second->m_initialized = true; "
   }
//    else //already exists, update from host if needed
//    {
//        //Do nothing for now, since writes to host deallocs device mem
//    }

   return result->second;
}





/*!
 *  \brief Update device with sparse matrix index contents that have an "size_t" type.
 *
 *  Update device with a SparseMatrix index range by specifying number of elements of "size_t" type.
 *  If sparse matrix does not have an allocation on the device for
 *  the current range, create a new allocation and if specified, also copy sparse matrix data to device.
 *  Saves newly allocated ranges to \p m_deviceMemIndexPointers_CL so sparse matrix can keep track of where
 *  and what it has stored on devices.
 *
 *  \param start Pointer to first index element in range to be updated with device.
 *  \param elems Number of (non-zero) index elements.
 *  \param device Pointer to the device that should be synched with.
 *  \param copy Boolean value that tells whether to only allocate or also copy sparse matrix index data to device. True copies, False only allocates.
 */
template <typename T>
typename SparseMatrix<T>::device_pointer_index_type_cl SparseMatrix<T>::updateDevice_Index_CL(size_t* start, size_t elems, backend::Device_CL* device, bool copy)
{
   DEBUG_TEXT_LEVEL3("SparseMatrix updating device CUDA\n")

   typename std::map<std::pair< cl_device_id, std::pair< size_t*, size_t > >, device_pointer_index_type_cl >::iterator result;

   std::pair< cl_device_id, std::pair< size_t*, size_t > > key(device->getDeviceID(), std::pair< size_t*, size_t >(start, elems));
   result = m_deviceMemIndexPointers_CL.find(key);

   if(result == m_deviceMemIndexPointers_CL.end()) //insert new, alloc mem and copy
   {
      device_pointer_index_type_cl temp = new backend::DeviceMemPointer_CL<size_t>(start, elems, device);
      if(copy)
      {
         //Make sure uptodate
         updateHost_CL();
         //Copy
         temp->copyHostToDevice();
      }
      result = m_deviceMemIndexPointers_CL.insert(m_deviceMemIndexPointers_CL.begin(), std::make_pair(key,temp));
   }
   else if(copy && !result->second->m_initialized) // we check for case when space is allocated but data was not copied, Multi-GPU case
   {
      //Make sure uptodate
      updateHost_CL(); // FIX IT: Only check for this copy and not for all copies.
      //Copy
      result->second->copyHostToDevice();	 // internally it will set "result->second->m_initialized = true; "
   }
//    else //already exists, update from host if needed
//    {
//        //Do nothing for now, since writes to host deallocs device mem
//    }

   return result->second;
}


/*!
 *  \brief Flushes the sparse matrix.
 *
 *  First it updates the sparse matrix from all its device allocations, then it releases all allocations.
 */
template <typename T>
void SparseMatrix<T>::flush_CL()
{
   DEBUG_TEXT_LEVEL3("SparseMatrix flush OpenCL\n")

   updateHost_CL();
   releaseDeviceAllocations_CL();
}

/*!
 *  \brief Updates the host from devices.
 *
 *  Updates the sparse matrix from all its device allocations.
 */
template <typename T>
inline void SparseMatrix<T>::updateHost_CL() const
{
   DEBUG_TEXT_LEVEL3("SparseMatrix updating host OpenCL\n")

   if(!m_deviceMemPointers_CL.empty())
   {
      typename std::map<std::pair< cl_device_id, std::pair< T*, size_t > >, device_pointer_type_cl >::const_iterator it;
      for(it = m_deviceMemPointers_CL.begin(); it != m_deviceMemPointers_CL.end(); ++it)
      {
         it->second->copyDeviceToHost();
      }
   }

   if(!m_deviceMemIndexPointers_CL.empty())
   {
      typename std::map<std::pair< cl_device_id, std::pair< size_t*, size_t > >, device_pointer_index_type_cl >::const_iterator it;
      for(it = m_deviceMemIndexPointers_CL.begin(); it != m_deviceMemIndexPointers_CL.end(); ++it)
      {
         it->second->copyDeviceToHost();
      }
   }
}

/*!
 *  \brief Invalidates the device data.
 *
 *  Invalidates the device data by releasing all allocations. This way the sparse matrix is updated
 *  and then data must be copied back to devices if used again.
 */
template <typename T>
inline void SparseMatrix<T>::invalidateDeviceData_CL()
{
   DEBUG_TEXT_LEVEL3("SparseMatrix invalidating device data OpenCL\n")

   //deallocs all device mem for matrix for now
   if(!m_deviceMemPointers_CL.empty() || !m_deviceMemIndexPointers_CL.empty())
   {
      releaseDeviceAllocations_CL();
   }
   //Could maybe be made better by only setting a flag that data is not valid
}

/*!
 *  \brief Releases device allocations.
 *
 *  Releases all device allocations for this sparse matrix. The memory pointers are removed.
 */
template <typename T>
inline void SparseMatrix<T>::releaseDeviceAllocations_CL()
{
   DEBUG_TEXT_LEVEL3("SparseMatrix releasing device allocations OpenCL\n")

   typename std::map<std::pair< cl_device_id, std::pair< T*, size_t > >, device_pointer_type_cl >::const_iterator it;
   for(it = m_deviceMemPointers_CL.begin(); it != m_deviceMemPointers_CL.end(); ++it)
   {
      delete it->second;
   }
   m_deviceMemPointers_CL.clear();


   typename std::map<std::pair< cl_device_id, std::pair< size_t*, size_t > >, device_pointer_index_type_cl >::const_iterator it2;
   for(it2 = m_deviceMemIndexPointers_CL.begin(); it2 != m_deviceMemIndexPointers_CL.end(); ++it2)
   {
      delete it2->second;
   }
   m_deviceMemIndexPointers_CL.clear();
}

}

#endif