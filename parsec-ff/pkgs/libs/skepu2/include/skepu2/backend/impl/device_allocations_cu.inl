namespace skepu2
{
namespace backend
{

/*!
 *  Static member initialization, keeps track of number of created instances.
 */
template <typename T>
DeviceAllocations_CU<T>* DeviceAllocations_CU<T>::instance = 0;

/*!
 *  Gets pointer to first instance, at first call a new instance is created.
 *
 *  \return Pointer to class instance.
 */
template <typename T>
DeviceAllocations_CU<T>* DeviceAllocations_CU<T>::getInstance()
{
   if(instance == 0)
   {
      instance = new DeviceAllocations_CU;
      _destroyer.SetDeviceAllocations_CU(instance);
   }
   return instance;
}

/*!
 *  The constructor initializes the devices.
 */
template <typename T>
DeviceAllocations_CU<T>::DeviceAllocations_CU()
{
	init();
}

/*!
 * Destructor
 */
template <typename T>
DeviceAllocations_CU<T>::~DeviceAllocations_CU()
{
}

/*!
 *  General initialization.
 */
template <typename T>
void DeviceAllocations_CU<T>::init()
{
   //srand(time(0));
}

/*!
 *  Add a device memory allocation to the list
 *  \param datapointer Pointer to device memory
 *  \param device_mem_pointer DeviceMemPointer_CU object related to the memory allocation
 *  \param deviceid ID of device where the allocation resides
 */
template <typename T>
void DeviceAllocations_CU<T>::addAllocation(void *datapointer,MemPointerBase *device_mem_pointer, int deviceid)
{
	std::pair<m_alloctype::iterator,bool> ret;
	ret = allocations[deviceid].insert(std::make_pair(datapointer,device_mem_pointer));
}

/*!
 *  Remove a device memory allocation from the list
 *  \param datapointer Pointer to device memory used to identify which object should be removed
 *  \param device_mem_pointer DeviceMemPointer_CU object related to the memory allocation
 *  \param deviceid ID of device where the allocation resides
 */
template <typename T>
void DeviceAllocations_CU<T>::removeAllocation(void *datapointer, MemPointerBase *device_mem_pointer, int deviceid)
{
	m_alloctype::iterator it;
	it = allocations[deviceid].find(datapointer);
	if (it != allocations[deviceid].end()) // check that the allocation has not allready been removed
	{
		// remove from list of allocations
		allocations[deviceid].erase(it);
		// invalidate device data
		it->second->markCopyInvalid();
		// free up device memory
		it->second->clearDevicePointer();
	}
}

/*!
 *  Attempts to free up memory on the device
 *  Called when memory allocation fails due to lack of memory
 *  \param minsize The amount of memory required by the failed malloc call
 *  \param deviceid The device to free up memory on
 */
template <typename T>
bool DeviceAllocations_CU<T>::freeAllocation(size_t minsize, int deviceid)
{
	m_alloctype::iterator it;
	size_t removedMem = 0;
	for (it=allocations[deviceid].begin(); it!=allocations[deviceid].end();++it)
	{
		if (removedMem < minsize)
			{
				removedMem += it->second->getMemSize();
				this->removeAllocation(it->first,it->second,deviceid);
			}
		else
			break;
	}
	return removedMem>=minsize;
}

/*!
 * Print information regarding device allocations
 * \param deviceid ID of device for which to print information
 */
template <typename T>
void DeviceAllocations_CU<T>::printList(int deviceid)
{
	m_alloctype::iterator it;
	int id=0;
	printf("-------------------\n");
	printf("printing list of allocations on device %d \n", deviceid);
	for (it=allocations[deviceid].begin(); it!=allocations[deviceid].end();++it)
	{
		printf("allocation   #%d \n",id);
		printf("mem pointer = %p \n",it->first);
		printf("size =        %d \n",it->second->getMemSize());
		id++;
	}
	printf("-------------------\n");
}
}

}

