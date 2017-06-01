#ifdef SKEPU_CUDA

#include <iostream>
#include <cuda.h>
#include <map>

#include "mem_pointer_base.h"

namespace skepu2
{
namespace backend
{

// Forward declaration
template <typename T>
class DeviceAllocations_CUDestroyer;
/*!
 * \class DeviceAllocations_CU
 *
 * singleton class keeping track of all skepu containers and their device memory allocations
*/

template <typename T>
class DeviceAllocations_CU
{
	typedef std::map<void *,MemPointerBase*> m_alloctype;
public:
	static DeviceAllocations_CU* getInstance();
	void addAllocation(void *datapointer,MemPointerBase *device_mem_pointer, int deviceid); 
	void removeAllocation(void *datapointer,MemPointerBase *device_mem_pointer, int deviceid);
	bool freeAllocation(size_t minsize,int deviceid);
	void printList(int deviceid); 
protected:
	DeviceAllocations_CU();

	virtual ~DeviceAllocations_CU();

	friend class DeviceAllocations_CUDestroyer<T>; // To safely clean resources
private:

	static DeviceAllocations_CUDestroyer<T> _destroyer;

	void init();

	m_alloctype allocations[MAX_GPU_DEVICES];	
  static DeviceAllocations_CU* instance;
};

template <typename T>
DeviceAllocations_CUDestroyer<T> DeviceAllocations_CU<T>::_destroyer;

template <typename T>
class DeviceAllocations_CUDestroyer
{
public:
   DeviceAllocations_CUDestroyer(DeviceAllocations_CU<T>* = 0);
   ~DeviceAllocations_CUDestroyer();

   void SetDeviceAllocations_CU(DeviceAllocations_CU<T>* s);
private:
   DeviceAllocations_CU<T>* _singleton;
};




template <typename T>
DeviceAllocations_CUDestroyer<T>::DeviceAllocations_CUDestroyer(DeviceAllocations_CU<T>* s)
{
   _singleton = s;
}

template <typename T>
DeviceAllocations_CUDestroyer<T>::~DeviceAllocations_CUDestroyer ()
{
   delete _singleton;
}

template <typename T>
void DeviceAllocations_CUDestroyer<T>::SetDeviceAllocations_CU (DeviceAllocations_CU<T>* s)
{
   _singleton = s;
}
}
}

#include "impl/device_allocations_cu.inl"


#endif
