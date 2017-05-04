/*! \file sparse_matrix_cu.inl
 *  \brief Contains the definitions of member functions of the SparseMatrix class related to \em CUDA backend.
 */

#ifdef SKEPU_CUDA

namespace skepu2
{

/*!
 *  \brief Update device with matrix content.
 *
 *  Update device with a SparseMatrix range by specifying array of non-zero elements.
 *  If SparseMatrix does not have an allocation on the device for
 *  the current range, create a new allocation and if specified, also copy SparseMatrix data to device.
 *  Saves newly allocated ranges to \p m_deviceMemPointers_CU so SparseMatrix can keep track of where
 *  and what it has stored on devices.
 *
 *  \param start Pointer to first (non-zero) element in range to be updated with device.
 *  \param elems Number of (non-zero) elements.
 *  \param deviceID Integer specififying the device that should be synched with.
 *  \param copy boolean value that tells whether to only allocate or also copy sparse
 *         matrix data to device. True copies, false only allocates.
 */
template <typename T>
typename SparseMatrix<T>::device_pointer_type_cu SparseMatrix<T>::updateDevice_CU(T* start, size_t elems, unsigned int deviceID, bool copy)
{
   DEBUG_TEXT_LEVEL3("SparseMatrix updating device CUDA\n")

   typename std::map<std::pair< unsigned int, std::pair< T*, size_t > >, device_pointer_type_cu >::iterator result;

   std::pair< unsigned int, std::pair< T*, size_t > > key(deviceID, std::pair< T*, size_t >(start, elems));

   result = m_deviceMemPointers_CU.find(key);

   if(result == m_deviceMemPointers_CU.end()) //insert new, alloc mem and copy
   {
      device_pointer_type_cu temp = new backend::DeviceMemPointer_CU<T>(start, elems, backend::Environment<int>::getInstance()->m_devices_CU.at(deviceID));
      if(copy)
      {
         //Make sure uptodate
         updateHost_CU();
         //Copy
         temp->copyHostToDevice();
      }
      result = m_deviceMemPointers_CU.insert(m_deviceMemPointers_CU.begin(), std::make_pair(key,temp));
   }
   //already exists, update from host if needed
   else if(copy && !result->second->isCopyValid()) // we check for case when space is allocated but data was not copied, Multi-GPU case
   {
      //Make sure uptodate
      updateHost_CU(); // FIX IT: Only check for this copy and not for all copies.
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
 *  Saves newly allocated ranges to \p m_deviceMemIndexPointers_CU so sparse matrix can keep track of where
 *  and what it has stored on devices.
 *
 *  \param start Pointer to first index element in range to be updated with device.
 *  \param elems Number of (non-zero) index elements.
 *  \param deviceID Integer specififying the device that should be synched with.
 *  \param copy Boolean value that tells whether to only allocate or also copy
 				sparse matrix index data to device. True copies, False only allocates.
 */
template <typename T>
typename SparseMatrix<T>::device_pointer_index_type_cu SparseMatrix<T>::updateDevice_Index_CU(size_t* start, size_t elems, unsigned int deviceID, bool copy)
{
   DEBUG_TEXT_LEVEL3("SparseMatrix updating device CUDA\n")

   typename std::map<std::pair< unsigned int, std::pair< size_t*, size_t > >, device_pointer_index_type_cu >::iterator result;

   std::pair< unsigned int, std::pair< size_t*, size_t > > key(deviceID, std::pair< size_t*, size_t >(start, elems));

   result = m_deviceMemIndexPointers_CU.find(key);

   if(result == m_deviceMemIndexPointers_CU.end()) //insert new, alloc mem and copy
   {
      device_pointer_index_type_cu temp = new backend::DeviceMemPointer_CU<size_t>(start, elems, backend::Environment<int>::getInstance()->m_devices_CU.at(deviceID), m_nameVerbose);
            
      if(copy)
      {
         //Make sure uptodate
         updateHost_CU();
         //Copy
         temp->copyHostToDevice();
      }
      result = m_deviceMemIndexPointers_CU.insert(m_deviceMemIndexPointers_CU.begin(), std::make_pair(key,temp));
   }
   //already exists, update from host if needed
   else if(copy && !result->second->isCopyValid()) // we check for case when space is allocated but data was not copied, Multi-GPU case
   {
      //Make sure uptodate
      updateHost_CU(); // FIX IT: Only check for this copy and not for all copies.
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
 *  \brief Flushes the matrix.
 *
 *  First it updates the matrix from all its device allocations, then it releases all allocations.
 */
template <typename T>
void SparseMatrix<T>::flush_CU()
{
   DEBUG_TEXT_LEVEL3("SparseMatrix flush CUDA\n")

   updateHost_CU();
   releaseDeviceAllocations_CU();
}

/*!
 *  \brief Updates the host from devices.
 *
 *  Updates the matrix from all its device allocations.
 */
template <typename T>
inline void SparseMatrix<T>::updateHost_CU() const
{
   DEBUG_TEXT_LEVEL3("SparseMatrix updating host CUDA\n")

   if(!m_deviceMemPointers_CU.empty())
   {
      typename std::map<std::pair< unsigned int, std::pair< T*, size_t > >, device_pointer_type_cu >::const_iterator it;
      for(it = m_deviceMemPointers_CU.begin(); it != m_deviceMemPointers_CU.end(); ++it)
      {
          if(it->second->isCopyValid())
            it->second->copyDeviceToHost();
      }
   }

   if(!m_deviceMemIndexPointers_CU.empty())
   {
      typename std::map<std::pair< unsigned int, std::pair< size_t*, size_t > >, device_pointer_index_type_cu >::const_iterator it;
      for(it = m_deviceMemIndexPointers_CU.begin(); it != m_deviceMemIndexPointers_CU.end(); ++it)
      {
          if(it->second->isCopyValid())
             it->second->copyDeviceToHost();
      }
   }
}

/*!
 *  \brief Invalidates the device data.
 *
 *  Invalidates the device data by releasing all allocations. This way the matrix is updated
 *  and then data must be copied back to devices if used again.
 */
template <typename T>
inline void SparseMatrix<T>::invalidateDeviceData_CU()
{
   DEBUG_TEXT_LEVEL3("SparseMatrix invalidating device data CUDA\n")

   //deallocs all device mem for SparseMatrix for now
   if(!m_deviceMemPointers_CU.empty() || !m_deviceMemIndexPointers_CU.empty())
   {
      releaseDeviceAllocations_CU();
   }

   //Could maybe be made better by only setting a flag that data is not valid
}

/*!
* Can be used to query whether sparse matrix is modified on a device or not.
*/
template <typename T>
bool SparseMatrix<T>::isModified_CU(unsigned int deviceID)
{
//    typename std::map<std::pair< unsigned int, std::pair< size_t*, size_t > >, device_pointer_index_type_cu >::iterator result;
//
//    std::pair< unsigned int, std::pair< size_t*, size_t > > key(deviceID, std::pair< size_t*, size_t >(start, elems));
//
//    result = m_deviceMemIndexPointers_CU.find(key);

//    return (result != m_deviceMemIndexPointers_CU[deviceID].end());
  return true;
}


/*!
 * Can be used to query whether matrix is already available on a device or not.
 */
template <typename T>
bool SparseMatrix<T>::isSparseMatrixOnDevice_CU(unsigned int deviceID)
{
//    typename std::map<std::pair< unsigned int, std::pair< size_t*, size_t > >, device_pointer_index_type_cu >::iterator result;
//
//    std::pair< unsigned int, std::pair< size_t*, size_t > > key(deviceID, std::pair< size_t*, size_t >(start, elems));
//
//    result = m_deviceMemIndexPointers_CU.find(key);
//
//    return (result != m_deviceMemIndexPointers_CU[deviceID].end());
  return true;
}

/*!
 *  \brief Releases device allocations.
 *
 *  Releases all device allocations for this SparseMatrix. The memory pointers are removed.
 */
template <typename T>
inline void SparseMatrix<T>::releaseDeviceAllocations_CU()
{
   DEBUG_TEXT_LEVEL3("SparseMatrix releasing device allocations CUDA\n")

   typename std::map<std::pair< unsigned int, std::pair< T*, size_t > >, device_pointer_type_cu >::const_iterator it;
   for(it = m_deviceMemPointers_CU.begin(); it != m_deviceMemPointers_CU.end(); ++it)
   {
      delete it->second;
   }
   m_deviceMemPointers_CU.clear();


   typename std::map<std::pair< unsigned int, std::pair< size_t*, size_t > >, device_pointer_index_type_cu >::const_iterator it2;
   for(it2 = m_deviceMemIndexPointers_CU.begin(); it2 != m_deviceMemIndexPointers_CU.end(); ++it2)
   {
      delete it2->second;
   }
   m_deviceMemIndexPointers_CU.clear();
}

}

#endif
