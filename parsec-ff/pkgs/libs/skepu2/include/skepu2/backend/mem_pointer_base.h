#ifndef MEMBASE
#define MEMBASE

namespace skepu2
{
namespace backend
{

class MemPointerBase
{
public:
	virtual size_t getMemSize() =0;
	virtual void markCopyInvalid() =0;
	virtual void clearDevicePointer() =0;
}; 

}
}

#endif

