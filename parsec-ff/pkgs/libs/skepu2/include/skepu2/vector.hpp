/*! \file vector.h
*  \brief Contains a class declaration for the Vector container.
 */

#ifndef VECTOR_H
#define VECTOR_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstddef>
#include <map>

#include "backend/malloc_allocator.h"

#ifdef SKEPU_PRECOMPILED

#include "backend/environment.h"
#include "backend/device_mem_pointer_cl.h"
#include "backend/device_mem_pointer_cu.h"

#endif // SKEPU_PRECOMPILED


namespace skepu2
{
	template<typename T>
	class Vector;
	
	// Proxy vector for user functions
	template<typename T>
	struct Vec
	{
		using ContainerType = Vector<T>;
		
		Vec(T *dataptr, size_t sizearg): data{dataptr}, size{sizearg} {} 
		Vec(): data{nullptr}, size{0} {} // empty proxy constructor
		
		T &operator[](size_t index)       { return this->data[index]; }
		T  operator[](size_t index) const { return this->data[index]; }
		
		T *data;
		size_t size;
	};
	
	template <typename T>
	class VectorIterator;
	
	/*!
	*  \class Vector
	*
	*  \brief A vector container class, implemented as a wrapper for std::vector.
	*
	*  A \p skepu::Vector is a container of vector/array type and is implemented as a wrapper for \p std::vector.
	*  Its interface and behaviour is largely compatible with \p std::vector but with some additions and variations.
	*  Instead of the regular element, it sometimes returns a proxy element so it can distinguish between reads
	*  and writes. It also keeps track of which parts of it are currently allocated and uploaded to the GPU.
	*  If a computation is done, changing the vector in the GPU memory, it is not directly transferred back to the
	*  host memory. Instead, the vector waits until an element is accessed before any copying is done.
	*
	*  It also implements support for allocating and de-allocating page-locked memory using cudaMallocHost and cudaFreeHost.
	*  This could help is running asynchronous operations especially when using multiple CUDA devices.
	*  It can be enabled by defining USE_PINNED_MEMORY flag in the skeleton program.
	*
	*  Please refer to C++ STL vector documentation for more information about CPU side implementation.
	*/
	template <typename T>
	class Vector
	{
		
	public:
		
		typedef typename std::vector<T>::size_type size_type;
		typedef T value_type;
		typedef ptrdiff_t difference_type;
		typedef T* pointer;
		typedef T& reference;
		typedef T const & const_reference;
		typedef Vec<T> proxy_type;
			
		typedef VectorIterator<T> iterator;
		typedef VectorIterator<const T> const_iterator;
		
		//-- For Testing --//
		
		friend std::ostream& operator<< (std::ostream& output, Vector<T>& vec)
		{
			for (typename Vector<T>::size_type i = 0; i < vec.size(); ++i)
			{
				output<<vec.at(i) <<" ";
			}
			return output;
		}
		
	public: //-- For Testing --//
		
		void randomize(int min = 0, int max = RAND_MAX);
		void save(const std::string& filename);
		void load(const std::string& filename, size_type numElements = 0);
		
	public: //-- Typedefs --//

#ifdef SKEPU_CUDA
		typedef backend::DeviceMemPointer_CU<T>* device_pointer_type_cu;
#endif

#ifdef SKEPU_OPENCL
		typedef backend::DeviceMemPointer_CL<T>* device_pointer_type_cl;
		typedef backend::DeviceMemPointer_CL<const T>* device_const_pointer_type_cl;
#endif

	public: //-- Constructors & Destructor --//
		
		Vector();
		
		Vector(const Vector& vec);
		Vector(Vector&& vec);
		Vector(std::initializer_list<T> l);
		explicit Vector(size_type num, const T& val = T());
		Vector(T * const ptr, size_type size, bool deallocEnabled = true);
		
		~Vector();
		
	public: //-- Member classes --//
		
		class proxy_elem;
		
	public: //-- Operators --//
		
		const T& operator[](const size_type index) const;
#ifdef SKEPU_PRECOMPILED
		proxy_elem operator[](const size_type index);
#else
		T& operator[](const size_type index);
#endif // SKEPU_PRECOMPILED
		
		
		Vec<T> hostProxy()
		{
			return {this->m_data, this->m_size};
		}
		
		Vector<T>& operator=(const Vector<T>&);
		
		bool operator==(const Vector<T>&);
		bool operator!=(const Vector<T>&);
		bool operator< (const Vector<T>&);
		bool operator> (const Vector<T>&);
		bool operator<=(const Vector<T>&);
		bool operator>=(const Vector<T>&);
		
	public: //-- STL vector regular interface --//
		
		// Iterators
		iterator begin();
		iterator end();
		
		const_iterator begin() const;
		const_iterator end() const;
		
		// Capacity
		size_type capacity() const   { return this->m_capacity; }
		size_type size() const       { return this->m_size; }
		size_type total_cols() const { return this->m_size; }
		size_type max_size() const   { return 1073741823; }

		void resize(size_type num, T val = T());
		bool empty() const;
		void reserve(size_type size);
		
		const Vector<T>& getParent() const { return *this; }
		Vector<T>& getParent() { return *this; }
		
		// Element access
		const T& at(size_type loc) const;
		const T& back() const;
		const T& front() const;
		
#ifdef SKEPU_PRECOMPILED
		proxy_elem at(size_type loc);
		proxy_elem back();
		proxy_elem front();
#else
		T& at(size_type loc);
		T& back();
		T& front();
#endif // SKEPU_PRECOMPILED
		
		// Modifiers
		void assign( size_type num, const T& val );
		
		template <typename input_iterator>
		void assign( input_iterator start, input_iterator end );
		void clear();
		
		iterator erase( iterator loc );
		iterator erase( iterator start, iterator end );
		iterator insert( iterator loc, const T& val );
		
		void insert( iterator loc, size_type num, const T& val );
		void pop_back();
		void push_back(const T& val);
		
		void swap(Vector<T>& from);
		
		T *getAddress() { return m_data; }
		
	public: //-- Additions to interface --//
		
#ifdef SKEPU_OPENCL
		device_const_pointer_type_cl updateDevice_CL(const T* start, size_type numElements, backend::Device_CL* device, bool copy) const;
		device_pointer_type_cl updateDevice_CL(T* start, size_type numElements, backend::Device_CL* device, bool copy);
		void flush_CL();
		bool isVectorOnDevice_CL(backend::Device_CL* device, bool multi=false) const;
#endif
		
#ifdef SKEPU_CUDA
		void copyDataToAnInvalidDeviceCopy(backend::DeviceMemPointer_CU<T> *copy, size_t deviceID, size_t streamID = 0) const;
		device_pointer_type_cu updateDevice_CU(T* start, size_type numElements, size_t deviceID, AccessMode accessMode, bool markOnlyLocalCopiesInvalid = false, size_t streamID = 0) const;
		void flush_CU();
		bool isVectorOnDevice_CU(size_t deviceID) const;
		bool isModified_CU(size_t deviceID) const;
		
		std::pair<device_pointer_type_cu, Vec<T>> cudaProxy(size_t deviceID, AccessMode accessMode)
		{
			device_pointer_type_cu devptr = this->updateDevice_CU(this->m_data, this->m_size, deviceID, accessMode);
			return {devptr, {devptr->getDeviceDataPointer(), this->m_size}};
		}
#endif
		
		void flush();
		
		// Does not care about device data, use with care
		T& operator()(const size_type index) { return m_data[index]; }
		const T& operator()(const size_type index) const { return m_data[index]; }
		
		// To be able to explicitly force updates without flushing entire vector.
		// Could be used with operator () above to avoid unneccesary function calls
		// due to implicit synch.
		void updateHost(bool = true) const;
		void invalidateDeviceData(bool = true) const;
		void updateHostAndInvalidateDevice();
		void releaseDeviceAllocations();
		void updateHostAndReleaseDeviceAllocations();
		
		void setValidFlag(bool val)
		{
			m_valid = val;
		}
		
	private: //-- Data --//
		T *m_data;
		mutable bool m_valid; /*! to keep track of whether the main copy is valid or not */
		size_type m_capacity;
		size_type m_size;
		bool m_deallocEnabled;
		mutable bool m_noValidDeviceCopy;

#ifdef SKEPU_OPENCL
		mutable std::map<std::pair<cl_device_id, const T* >, device_pointer_type_cl > m_deviceMemPointers_CL;
		mutable std::map<std::pair<cl_device_id, const T* >, device_const_pointer_type_cl > m_deviceConstMemPointers_CL;
#endif

#ifdef SKEPU_CUDA
//      std::map<std::pair< int, std::pair< T*, size_type > >, device_pointer_type_cu > m_deviceMemPointers_CU;
		mutable std::map<std::pair< T*, size_type >, device_pointer_type_cu > m_deviceMemPointers_CU[MAX_GPU_DEVICES];

   /*! This is a temporary list that keeps track of copies that are changed on device but are not synced with host memory... */
		mutable std::map<std::pair< T*, size_type >, device_pointer_type_cu > m_deviceMemPointers_Modified_CU[MAX_GPU_DEVICES];
#endif

//-- Private helpers --//

#ifdef SKEPU_OPENCL
		void updateHost_CL() const;
		void invalidateDeviceData_CL() const;
		void releaseDeviceAllocations_CL() const;
#endif

#ifdef SKEPU_CUDA
		void updateHost_CU(int deviceID = -1) const;
		void invalidateDeviceData_CU(int deviceID = -1) const;
		void releaseDeviceAllocations_CU(int deviceID = -1) const;
#endif
		
	};

	/*!
	*  \class Vector::iterator
	*  \author Johan Enmyren, Usman Dastgeer
	*  \version 0.7
	*
	*  \brief An vector iterator class.
	*
	*  An iterator class for \p skepu::Vector. behaves like the vector iterator for \p std::vector
	*  but similar to \p skepu::Vector it sometimes returns a \p proxy_elem instead of the actual
	*  element. Also makes sure the vector is properly synchronized with device before returning
	*  any elements.
	*/
	template <typename T>
	class VectorIterator : public std::iterator<std::random_access_iterator_tag, T>
	{
	public:
		typedef VectorIterator<T> iterator;
		typedef VectorIterator<const T> const_iterator;
	   typedef typename std::conditional<std::is_const<T>::value,
						const Vector<typename std::remove_const<T>::type>, Vector<T>>::type parent_type;
		
		typedef Vec<T> proxy_type;
	
	public: //-- Constructors & Destructor --//
		
		VectorIterator(parent_type& vec, T *std_iterator);
		
	public: //-- Types --//
		
#ifdef SKEPU_CUDA
		typedef typename parent_type::device_pointer_type_cu device_pointer_type_cu;
#endif
		
#ifdef SKEPU_OPENCL
		typedef typename parent_type::device_pointer_type_cl device_pointer_type_cl;
#endif
		
	public: //-- Extras --//
		
		Index1D getIndex() const;
		
		parent_type& getParent() const;
		iterator& begin(); // returns itself
		size_t size(); // returns number of elements "left" in parent container from this index
		
		T* getAddress() const;
		
		Vec<T> hostProxy() { return {this->m_std_iterator, this->size()}; }
		
		// Does not care about device data, use with care...sometimes pass negative indices...
		T& operator()(const ssize_t index = 0);
		const T& operator()(const ssize_t index) const;
		
	public: //-- Operators --//
		
		T& operator[](const ssize_t index);
		const T& operator[](const ssize_t index) const;
		
		bool operator==(const iterator& i);
		bool operator!=(const iterator& i);
		bool operator<(const iterator& i);
		bool operator>(const iterator& i);
		bool operator<=(const iterator& i);
		bool operator>=(const iterator& i);
		
		const iterator& operator++();
		iterator operator++(int);
		const iterator& operator--();
		iterator operator--(int);
		
		const iterator& operator+=(const ssize_t i);
		const iterator& operator-=(const ssize_t i);
		
		iterator operator-(const ssize_t i) const;
		iterator operator+(const ssize_t i) const;
		
		typename parent_type::difference_type operator-(const iterator& i) const;
		
		T& operator *();
		const T& operator* () const;
		
		const T& operator-> () const;
		T& operator-> ();
		
	private: //-- Data --//
		
		parent_type& m_parent;
		T *m_std_iterator;
	};
	
}

#include "backend/impl/vector/vector.inl"

#endif
