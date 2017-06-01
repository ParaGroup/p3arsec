/*!
 *  \ingroup helpers
 */

/*!
 *  \class malloc_allocator
 *
 *  \brief A custom memory allocator used with \p std::vector.
 *
 *
 *  It is a custom memory allocator (CUDA PINNED memory allocation) that is used with STL
 *  vector container used within SkePU containers to allow overlapping
 *  communication and computation with CUDA calls. It allocates memory using cudaMallocHost
 *  and de-allocate it using cudaHostFree.
*/

#ifndef MALLOC_ALLOCATOR_H
#define MALLOC_ALLOCATOR_H

#include "debug.h"

#ifdef USE_PINNED_MEMORY
#include <cuda.h>
#endif

namespace skepu2
{
namespace backend
{

template <class T>
class malloc_allocator
{
public:
   typedef T                 value_type;
   typedef value_type*       pointer;
   typedef const value_type* const_pointer;
   typedef value_type&       reference;
   typedef const value_type& const_reference;
   typedef std::size_t       size_type;
   typedef std::ptrdiff_t    difference_type;

private:
   void operator=(const malloc_allocator&);
   static int count;

public:
   malloc_allocator() {}
   malloc_allocator(const malloc_allocator&) {}
   ~malloc_allocator() {}


public:
   template <class U>
   malloc_allocator(const malloc_allocator<U>&) {}

   template <class U>
   struct rebind
   {
      typedef malloc_allocator<U> other;
   };

   pointer address(reference x) const
   {
      return &x;
   }

   void construct(pointer p, const value_type& x)
   {
      new(p) value_type(x);
   }

   void destroy(pointer p)
   {
      p->~value_type();
   }

   const_pointer address(const_reference x) const
   {
      return &x;
   }

   pointer allocate(size_type n, const_pointer = 0)
   {
      void* p=0;

#ifdef USE_PINNED_MEMORY
      cudaError_t err;
      DEBUG_TEXT_LEVEL1("Pinned memory allocation ******\n")
//		err = cudaMallocHost((void **)&p, n * sizeof(T));	// Works before CUDA 4.0
      err = cudaHostAlloc((void **)&p, n * sizeof(T), cudaHostAllocPortable); // For CUDA 4.0
//		err = cudaHostAlloc((void **)&p, n * sizeof(T)); // For CUDA 4.0

      if(err)
      {
         SKEPU_ERROR("*** Error Malloc: " + cudaGetErrorString(err));
      }
#else
      p = std::malloc(n * sizeof(T));
#endif

      if (!p)
      {
         SKEPU_ERROR("*** Error Malloc **************");
      }
      return static_cast<pointer>(p);
   }

   void deallocate(pointer p, size_type)
   {
#ifdef USE_PINNED_MEMORY
      cudaError_t err;
      err = cudaFreeHost(p);
      if(err)
      {
         SKEPU_ERROR("*** Error De-Alloc: " + cudaGetErrorString(err));
      }
      else
      {
         DEBUG_TEXT_LEVEL1("Pinned memory de-allocation Successful ******\n")
      }
#else
      std::free(p);
#endif
   }

   size_type max_size() const
   {
      return static_cast<size_type>(-1) / sizeof(value_type);
   }
};

template <class T>
int malloc_allocator<T>::count = 0;

template <class T>
inline bool operator==(const malloc_allocator<T>&, const malloc_allocator<T>&)
{
   return true;
}

template <class T>
inline bool operator!=(const malloc_allocator<T>&, const malloc_allocator<T>&)
{
   return false;
}


template<>
class malloc_allocator<void>
{
   typedef void        value_type;
   typedef void*       pointer;
   typedef const void* const_pointer;

   template <class U>
   struct rebind
   {
      typedef malloc_allocator<U> other;
   };
};

}
}

#endif


