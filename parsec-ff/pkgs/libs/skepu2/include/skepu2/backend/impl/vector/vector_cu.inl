#ifdef SKEPU_CUDA

namespace skepu2
{

/*!
 * \brief Used by updateDevice_CU function to copy data to a device copy..
 * the device copy could be a new one (just created) or an existing one with stale (marked invalid) data
 * it tries to copy data from copies in existing device memory, then from host memory and in the end
 * from other device memories... it can partially copy data from different sources in the process
 * \param copy it is the actual copy that the data is written to...
 * \param deviceID id of the device where this copy belongs...
 * \param streamID id of the CUDA Stream that will use this copy when using MultiStream (will be used if USE_PINNED_MEMORY is defined)
 */
template <typename T>
void Vector<T>::copyDataToAnInvalidDeviceCopy(backend::DeviceMemPointer_CU<T> *copy, size_t deviceID, size_t streamID) const
{
   typename std::map<std::pair< T*,size_type >, device_pointer_type_cu >::const_iterator it;
   
   assert(copy->isCopyValid() == false && copy->m_numOfRanges == 1);
   if(copy->deviceDataHasChanged())
   {
      std::cerr << copy->m_nameVerbose << " on GPU_" << deviceID << " is invalid but changed???\n";
   }
   assert(copy->deviceDataHasChanged() == false);
   

   backend::UpdateInf<T> updateStruct[MAX_COPYINF_SIZE];
   size_type sizeUpdStr = 0;

   /*!
      * first check for copies within same device that is overlapping and valid...
      * yes, there could be >1 copies,
      * e.g.,
      * 2 overlapping "valid" copies if none of them is modified...
      * 2 non-overlapping "valid" copies if atleast one of them is written...
      */
   if(m_deviceMemPointers_CU[deviceID].empty() == false)
   {
      for(it = m_deviceMemPointers_CU[deviceID].begin(); it != m_deviceMemPointers_CU[deviceID].end(); ++it)
      {
         if(it->second != copy && it->second->isCopyValid() && it->second->doCopiesOverlap(copy))
         {
            copy->copiesOverlapInf(it->second, updateStruct, sizeUpdStr); /*! sizeUpdStr is passed by referece.. will be updated inside called function */

            if(copy->m_numOfRanges == 0) // if already found required data then leave...
               break;
//                      if(it->second->deviceDataHasChanged())
            {
               /*! if "src" copy has modified contents then copy those contents to current "dst" copy but keep modified flag set for "src" copy
                  ** now, if u read "dst" copy then no problem, "src" copy has "modified" flag, dst has no such flag
                  *  if host or other copies need contents, they get from "src"
                  ** but if u write "dst" copy then also no problem as later code in this function will mark "src" as invalid copy as "dst" has latest modified
                  * contents. now if host or other need to copy data they can copy from "dst"
                  *keep that  */
            }
            /*! At one point in time, there could be at >one valid copy per each device for a container */
         }
      }
   }

   /*! if still there exist some parts (ranges) in copy that cannot be found in valid copies present in current device memory... */
   if(copy->m_numOfRanges != 0)
   {
      if(m_valid) /*! if main copy (Host) is valid then copy from there as copying from other device memories' valid copies wont be much faster than HTD? */
      {
         copy->copyInfFromHostToDevice(updateStruct, sizeUpdStr); /*! sizeUpdStr is passed by referece.. will be updated inside called function */
         assert(copy->m_numOfRanges == 0);
      }
      else /*! unfortunately, main copy is invalid so need to look for copies in other device memories... */
      {
         backend::Environment<int> *env = backend::Environment<int>::getInstance();
         if(env->m_peerAccessEnabled) /*! if peer acces enabled for all of them then satt Bismillah, i.e. can transfer directly from other GPUs copies... */
         {
            /*! Copies from valid overlapping copies in other device mmeories **/
            for(int devID = 0; devID < MAX_GPU_DEVICES; ++devID)
            {
               if(deviceID == devID || m_deviceMemPointers_CU[devID].empty())
                  continue;

               for(it = m_deviceMemPointers_CU[devID].begin(); it != m_deviceMemPointers_CU[devID].end(); ++it)
               {
                  if(it->second->isCopyValid() && it->second->doCopiesOverlap(copy))
                  {
                     copy->copiesOverlapInf(it->second, updateStruct, sizeUpdStr); /*! sizeUpdStr is passed by referece.. will be updated inside called function */
                     if(copy->m_numOfRanges == 0) // if found all copies then go back to out label...
                        goto out_label;
                  }
               }
            }
out_label:
            /*! it is posible that some parts are not copied yet (not present in device memories) so can copy them from host */
            if(copy->m_numOfRanges > 0)
               copy->copyInfFromHostToDevice(updateStruct, sizeUpdStr); /*! sizeUpdStr is passed by referece.. will be updated inside called function */
               
            assert(copy->m_numOfRanges == 0);
         }
         /*!
            * if peer access is not enabled then copy all overlapping "modified" copies from other device memories to host main copy and then copy it from there
            * it does not guarantee that the main copy is valid as there might be some nonoverlapping copies in current or other devices that are modified
            * and not written back to main copy but atleast it ensures that its safe to copy overlapping parts
            */
         else
         {
            /*! Copies all overlapping copies from other devices back to host and mark them as invalid **/
            for(int devID = 0; devID < MAX_GPU_DEVICES; ++devID)
            {
               if(deviceID == devID || m_deviceMemPointers_CU[devID].empty())
                  continue;

               for(it = m_deviceMemPointers_CU[devID].begin(); it != m_deviceMemPointers_CU[devID].end(); ++it)
               {
                  if(it->second->isCopyValid() && it->second->deviceDataHasChanged() &&  it->second->doCopiesOverlap(copy))
                  {
                     it->second->copyDeviceToHost(); /*! first copy it back to host... internally set modified flag to false */

                     /*! remove this copy now from list of copies to be updated back to host */
                     assert(m_deviceMemPointers_Modified_CU[devID].find(it->first) != m_deviceMemPointers_Modified_CU[devID].end());
                     m_deviceMemPointers_Modified_CU[devID].erase(m_deviceMemPointers_Modified_CU[devID].find(it->first));

                     /*! stupid condition as copy is not updated inside this loop, m_numOfRanges is not modified here */
//                      if(copy->m_numOfRanges == 0) // if found all copies then go back to out label...
//                         goto out_label_2;
                  }
               }
            }
// out_label_2: /*! now copy from host memory... the m_valid flag could be false at this time... updates later in function... */
            copy->copyInfFromHostToDevice(updateStruct, sizeUpdStr); /*! sizeUpdStr is passed by referece.. will be updated inside called function */
            assert(copy->m_numOfRanges == 0);
         }
      }
   }

   /*! now do actual Copy from all possible sources, HTD and DTD from within same and from other devices...
   * internally sets the m_valid flag for this copy
   */
   copy->copyAllRangesToDevice(updateStruct, sizeUpdStr, streamID);

   /*! reset ranges to default range which is total copy */
   copy->resetRanges();
}



/*!
 *  \brief Update device with vector content.
 *
 *  Update device with a vector range. If vector does not have an allocation on the device for
 *  the current range, create a new allocation and if specified, also copy vector data to device.
 *  Saves newly allocated ranges to \p m_deviceMemPointers_CU so vector can keep track of where
 *  and what it has stored on devices.
 *
 *  \param start Pointer to first element in range to be updated with device.
 *  \param numElements Number of elemets in range.
 *  \param deviceID Integer specififying the device that should be synched with.
 *  \param copy Boolean value that tells whether to only allocate or also copy vector data to device. True copies, False only allocates.
 *  \param writeAccess specifies whether this copy is going to be read or written...
 *  \param markOnlyLocalCopiesInvalid This is for optimizations in multi-GPU execution, passed to true to only mark parent and local copies within that device memory as invalid...
 *  \param streamID id of the CUDA Stream that will be using this vector range when using MultiStream (define USE_MULTI_STREAM and USE_PINNED_MEMORY)
 */
template <typename T>
typename Vector<T>::device_pointer_type_cu Vector<T>::updateDevice_CU(T* start, size_type numElements, size_t deviceID, AccessMode accessMode, bool markOnlyLocalCopiesInvalid, size_t streamID) const
{
    DEBUG_TEXT_LEVEL3("Vector updating device CUDA\n")
    
   /// \p m_noValidDeviceCopy is an optimization flag which is true when there is no valid device copy...
   /// used to skip invalidDeviceCopy function call just like updateHost() is only called when m_valid is not set
   if(m_noValidDeviceCopy)
      m_noValidDeviceCopy = false;
   
   typename std::map<std::pair< T*, size_type>, device_pointer_type_cu >::iterator result;
   std::pair< T*, size_type > key(start, numElements);
   typename std::map<std::pair< T*,size_type >, device_pointer_type_cu >::const_iterator it;

   result = m_deviceMemPointers_CU[deviceID].find(key);

   device_pointer_type_cu tempCopy = NULL;
   
#ifdef NO_LAZY_MEMORY_COPYING
   if(result == m_deviceMemPointers_CU[deviceID].end()) //no copy should be reused, right? may be not in multigpu case as it is allocated and then copied data
   {
      tempCopy = new backend::DeviceMemPointer_CU<T>(start, numElements, backend::Environment<int>::getInstance()->m_devices_CU.at(deviceID), "noname");
      if(hasReadAccess(accessMode))
      {
         tempCopy->copyHostToDevice();
      }
      result = m_deviceMemPointers_CU[deviceID].insert(m_deviceMemPointers_CU[deviceID].begin(), std::make_pair(key,tempCopy));
   }
   else
   {
      tempCopy = result->second;

      //already exists but need to update contents...
      if(hasReadAccess(accessMode) && tempCopy->isCopyValid() == false) // we check whether the copy is invalid and we need to copy data
      {
         tempCopy->copyHostToDevice();
      }
   }
   return result->second;
#else
   if(result == m_deviceMemPointers_CU[deviceID].end()) //insert new, alloc mem and copy
   {
      tempCopy = new backend::DeviceMemPointer_CU<T>(start, numElements, backend::Environment<int>::getInstance()->m_devices_CU.at(deviceID), "noname");

      if(hasReadAccess(accessMode))
      {
         copyDataToAnInvalidDeviceCopy(tempCopy, deviceID, streamID);
      }
      result = m_deviceMemPointers_CU[deviceID].insert(m_deviceMemPointers_CU[deviceID].begin(), std::make_pair(key,tempCopy));
   }
   else
   {
      tempCopy = result->second;

      //already exists but need to update contents...
      if(hasReadAccess(accessMode) && tempCopy->isCopyValid() == false) // we check whether the copy is invalid and we need to copy data
      {
         copyDataToAnInvalidDeviceCopy(tempCopy, deviceID, streamID);
      }
   }


   /*! TODO: BEFORE returning, MARK all other copies as invalid if you are writing this copy and they are overlapping with this copy */
   assert(tempCopy != NULL);
   if(hasWriteAccess(accessMode))
   {
      /*! add this copy to modified list... This list keeps track of copies that have modified data which is not written back.. so far */
      m_deviceMemPointers_Modified_CU[deviceID].insert(m_deviceMemPointers_Modified_CU[deviceID].begin(), std::make_pair(key,tempCopy));
      
      /*! First, mark parent copy invalid... */
      m_valid = false;
      
//       markOnlyLocalCopiesInvalid = false;
      // mark only local copies invalid... each device will do that in multi-gpu execution for single call. in the end they will set parent copy as invalid....
      if(markOnlyLocalCopiesInvalid) 
      {
         if(m_deviceMemPointers_CU[deviceID].empty()) // no copies...
            return result->second;

         for(it = m_deviceMemPointers_CU[deviceID].begin(); it != m_deviceMemPointers_CU[deviceID].end(); ++it)
         {
            if(tempCopy != it->second && it->second->isCopyValid() && tempCopy->doCopiesOverlap(it->second, true))
            {
               /*!
                * this is possible considering gpu-gpu transfers and in some other cases
                * e.g. map(v1 RW); ... map2(..., v1 Written);
                */
               if(it->second->deviceDataHasChanged())
               {
//                      assert(false); /*! TODO: fix this */
                  /*!
                   * if not fully overlapped then need to transfer as some data should be written back to device memory
                   * if fully overlapped then no need to update it as it is overwritten in current copy...
                   */
                  if(it->second->doOverlapAndCoverFully(tempCopy) == false)
                  {
                     it->second->copyDeviceToHost();
                  }

                  /*! should delete this copy from this list as it needs not to be updated back... */
                  assert(m_deviceMemPointers_Modified_CU[deviceID].find(it->first) != m_deviceMemPointers_Modified_CU[deviceID].end());
                  m_deviceMemPointers_Modified_CU[deviceID].erase(m_deviceMemPointers_Modified_CU[deviceID].find(it->first));
               }

               /*! mark copy invalid */
               it->second->markCopyInvalid();
            }
         }
      }
      else
      {
         /*! TODO: Mark all overlapping copies from all devices as invalid **/
         for(int devID = 0; devID < MAX_GPU_DEVICES; ++devID)
         {
            if(m_deviceMemPointers_CU[devID].empty()) // no copies...
               continue;

            for(it = m_deviceMemPointers_CU[devID].begin(); it != m_deviceMemPointers_CU[devID].end(); ++it)
            {
               if(tempCopy != it->second && it->second->isCopyValid() && tempCopy->doCopiesOverlap(it->second, true))
               {
                  /*!
                  * this is possible considering gpu-gpu transfers and in some other cases
                  * e.g. map(v1 RW); ... map2(..., v1 Written);
                  */
                  if(it->second->deviceDataHasChanged())
                  {
   //                      assert(false); /*! TODO: fix this */
                     /*!
                     * if not fully overlapped then need to transfer as some data should be written back to device memory
                     * if fully overlapped then no need to update it as it is overwritten in current copy...
                     */
                     if(it->second->doOverlapAndCoverFully(tempCopy) == false)
                     {
                        it->second->copyDeviceToHost();
                     }

                     /*! should delete this copy from this list as it needs not to be updated back... */
                     assert(m_deviceMemPointers_Modified_CU[devID].find(it->first) != m_deviceMemPointers_Modified_CU[devID].end());
                     m_deviceMemPointers_Modified_CU[devID].erase(m_deviceMemPointers_Modified_CU[devID].find(it->first));
                  }

                  /*! mark copy invalid */
                  it->second->markCopyInvalid();
               }
            }
         }
      }
   }
   /*! TODO: Update main copy valid flag... set it to "true", i.e., valid, if there exist no modified device copy **/
   else
   {
      bool noModifiedDevCopy = true;
      /*! Check all overlapping copies from all devices as invalid **/
      for(int devID = 0; devID < MAX_GPU_DEVICES; ++devID)
      {
         if (m_deviceMemPointers_Modified_CU[devID].empty() == false)
         {
//             for(it = m_deviceMemPointers_Modified_CU[devID].begin(); it != m_deviceMemPointers_Modified_CU[devID].end(); ++it)
//             {
//                std::cerr << m_nameVerbose << ", GPU_" << devID << ", size: " << it->second->m_numElements
//                          << (it->second->isCopyValid()? " Valid, ": " *NOT* VALID, ") 
//                          << (it->second->deviceDataHasChanged()? " Changed ": " *NOT* Changed ")
//                          << "\n";
//             }
            noModifiedDevCopy = false;
            break;
         }
      }
      if(noModifiedDevCopy)
         m_valid = true;
      else
         m_valid = false;
   }

   return result->second;
#endif   
}

/*!
 * Can be used to query whether vector is already available on a device or not.
 */
template <typename T>
bool Vector<T>::isVectorOnDevice_CU(size_t deviceID) const
{
   typename std::map<std::pair< T*,size_type >, device_pointer_type_cu >::iterator result;
   std::pair< T*, size_type > key(&m_data[0], m_size);

   result = m_deviceMemPointers_CU[deviceID].find(key);
   return !(result == m_deviceMemPointers_CU[deviceID].end());
}

/*!
 *  \brief Flushes the vector.
 *
 *  First it updates the vector from all its device allocations, then it deletes all allocations.
 */
template <typename T>
void Vector<T>::flush_CU()
{
   updateHost_CU();
   releaseDeviceAllocations_CU();
}

/*!
 *  \brief Updates the host from devices.
 *
 *  Updates the vector from all its device allocations.
 */
template <typename T>
inline void Vector<T>::updateHost_CU(int deviceID) const
{
   typename std::map<std::pair< T*,size_type >, device_pointer_type_cu >::const_iterator it;

#ifdef NO_LAZY_MEMORY_COPYING
   for(deviceID = 0; deviceID < MAX_GPU_DEVICES; ++deviceID)
   {
      if(m_deviceMemPointers_CU[deviceID].empty())
      {
         continue;
      }

      for(it = m_deviceMemPointers_CU[deviceID].begin(); it != m_deviceMemPointers_CU[deviceID].end(); ++it)
      {
         if(it->second->deviceDataHasChanged())
         {
            /*! At one point in time, there could >1 valid "modified" copy per each device for a container */
            it->second->copyDeviceToHost();
         }
      }
   }
   backend::Environment<int>::getInstance()->finishAll_CU(0, MAX_GPU_DEVICES+1);
   return;
#endif

   if(deviceID < 0) // do it for all devices....
   {
      for(deviceID = 0; deviceID < MAX_GPU_DEVICES; ++deviceID)
      {
         if(m_deviceMemPointers_Modified_CU[deviceID].empty())
         {
            continue;
         }

         for(it = m_deviceMemPointers_Modified_CU[deviceID].begin(); it != m_deviceMemPointers_Modified_CU[deviceID].end(); ++it)
         {
            assert(it->second->isCopyValid() && it->second->deviceDataHasChanged());
            {
               /*! At one point in time, there could >1 valid "modified" copy per each device for a container */
               it->second->copyDeviceToHost();
            }
         }
         m_deviceMemPointers_Modified_CU[deviceID].clear();
         backend::Environment<int>::getInstance()->finishAll_CU(deviceID, deviceID+1);
      }
   }
   else if(!m_deviceMemPointers_Modified_CU[deviceID].empty())
   {
      for(it = m_deviceMemPointers_Modified_CU[deviceID].begin(); it != m_deviceMemPointers_Modified_CU[deviceID].end(); ++it)
      {
         assert(it->second->isCopyValid() && it->second->deviceDataHasChanged());
         {
            /*! At one point in time, there could >1 valid "modified" copy per each device for a container */
            it->second->copyDeviceToHost();
         }
      }
      m_deviceMemPointers_Modified_CU[deviceID].clear();
      backend::Environment<int>::getInstance()->finishAll_CU(deviceID, deviceID+1);
   }
}


/*!
* Can be used to query whether vector is modified on a device or not.
*/
template <typename T>
bool Vector<T>::isModified_CU(size_t deviceID) const
{
   typename std::map<std::pair< T*,size_type >, device_pointer_type_cu >::const_iterator result;
   std::pair< T*, size_type > key(&m_data[0], size());

   result = m_deviceMemPointers_CU[deviceID].find(key);
   return (result != m_deviceMemPointers_CU[deviceID].end() && result->second->isCopyValid() && result->second->deviceDataHasChanged());
}

/*!
 *  \brief Invalidates the device data.
 *
 *  Invalidates the device data by marking all copies on one or more devices as invalid.
 *  This way the copies of device(s) are invalidated and data must be copied back to device copies before reading in future.
 */
template <typename T>
inline void Vector<T>::invalidateDeviceData_CU(int deviceID) const
{
   typename std::map<std::pair< T*,size_type >, device_pointer_type_cu >::const_iterator it;
   
   //deallocs all device mem for vector for now
   if(deviceID < 0) // do it for all devices....
   {
      for(deviceID = 0; deviceID < MAX_GPU_DEVICES; ++deviceID)
      {
         if(m_deviceMemPointers_CU[deviceID].empty())
            continue;

         for(it = m_deviceMemPointers_CU[deviceID].begin(); it != m_deviceMemPointers_CU[deviceID].end(); ++it)
         {
            it->second->markCopyInvalid();
         }
         m_deviceMemPointers_Modified_CU[deviceID].clear();
      }
   }
   else if(!m_deviceMemPointers_CU[deviceID].empty())
   {
      for(it = m_deviceMemPointers_CU[deviceID].begin(); it != m_deviceMemPointers_CU[deviceID].end(); ++it)
      {
         it->second->markCopyInvalid();
      }
      m_deviceMemPointers_Modified_CU[deviceID].clear();
   }
}

/*!
 *  \brief Releases device allocations.
 *
 *  Releases all device allocations for this vector. The memory pointers are removed.
 */
template <typename T>
inline void Vector<T>::releaseDeviceAllocations_CU(int deviceID) const
{
   typename std::map<std::pair< T*,size_type >, device_pointer_type_cu >::const_iterator it;

   if(deviceID < 0) // do it for all devices....
   {
      for(deviceID = 0; deviceID < MAX_GPU_DEVICES; ++deviceID)
      {
         if(m_deviceMemPointers_CU[deviceID].empty())
            continue;

         for(it = m_deviceMemPointers_CU[deviceID].begin(); it != m_deviceMemPointers_CU[deviceID].end(); ++it)
         {
            delete it->second;
         }
         m_deviceMemPointers_CU[deviceID].clear();
         m_deviceMemPointers_Modified_CU[deviceID].clear();
      }
   }
   else
   {
      for(it = m_deviceMemPointers_CU[deviceID].begin(); it != m_deviceMemPointers_CU[deviceID].end(); ++it)
      {
         delete it->second;
      }
      m_deviceMemPointers_CU[deviceID].clear();
      m_deviceMemPointers_Modified_CU[deviceID].clear();
   }
}

}

#endif


