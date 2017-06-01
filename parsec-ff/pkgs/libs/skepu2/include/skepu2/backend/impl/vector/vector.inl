namespace skepu2
{
	/*!
	 *  \brief Randomizes the vector.
	 *
	 *  Sets each element of the vector to a random number between \p min and \p max.
	 *  The numbers are generated as \p integers but are cast to the type of the vector.
	 *
	 *  \param min The smallest number an element can become.
	 *  \param max The largest number an element can become.
	 */
	template<typename T>
	void Vector<T>::randomize(int min, int max)
	{
		invalidateDeviceData();
		
		for (size_type i = 0; i < size(); i++)
		{
			m_data[i] = (T)( rand() % max + min );
		}
	}
	
	
	/*!
	 *  \brief Saves content of vector to a file.
	 *
	 *  Outputs the vector as text on one line with space between elements to the specified file.
	 *  Mainly for testing purposes.
	 *
	 *  \param filename Name of file to save to.
	 */
	template<typename T>
	void Vector<T>::save(const std::string& filename)
	{
		std::ofstream file(filename.c_str());
		
		if (file.is_open())
		{
			for (size_type i = 0; i < size(); ++i)
			{
				file << at(i) << " ";
			}
			file.close();
		}
		else
		{
			std::cout<<"Unable to open file\n";
		}
	}
	
	
	/*!
	 *  \brief Loads the vector from a file.
	 *
	 *  Reads a variable number of elements from a file. In the file, all elemets should be in ASCII
	 *  on one line with whitespace between each element. Mainly for testing purposes.
	 *
	 *  \param filename Name of file to save to.
	 *  \param numElements The number of elements to load. Default value 0 means all values.
	 */
	template<typename T>
	void Vector<T>::load(const std::string& filename, size_type numElements)
	{
		std::ifstream file(filename.c_str());

		if (file.is_open())
		{
			std::string line;
			getline (file,line);
			std::istringstream ss(line);
			T num;
			clear();
			
			// Load all elements
			if (numElements == 0)
			{
				while (ss >> num)
					push_back(num);
			}
			// Load only numElements elements
			else
			{
				for (size_type i = 0; i < numElements; ++i)
				{
					ss >> num;
					push_back(num);
				}
			}
			
			file.close();
		}
		else
		{
			std::cout<<"Unable to open file\n";
		}
	}
	
///////////////////////////////////////////////
// Constructors START
///////////////////////////////////////////////
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	inline Vector<T>::Vector(): m_capacity(10), m_size(0), m_deallocEnabled(true), m_valid(true), m_noValidDeviceCopy(true)
	{
		backend::allocateHostMemory<T>(m_data, m_capacity);
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector. The copy occurs w.r.t. elements.
	 *  As copy constructor creates a new storage.
	 *
	 *  Updates vector \p c before copying.
	 */
	template <typename T>
	inline Vector<T>::Vector(const Vector& c): m_capacity(c.m_capacity), m_size(c.m_size), m_deallocEnabled(true), m_valid(true), m_noValidDeviceCopy(true)
	{
		if (m_size < 1)
			SKEPU_ERROR("The vector size should be positive.");
		
		backend::allocateHostMemory<T>(m_data, m_capacity);
		c.updateHost();
		std::copy(c.m_data, c.m_data + m_size, m_data);
	}
	
	
	template <typename T>
	inline Vector<T>::Vector(Vector&& c):
		m_data(nullptr),
		m_capacity(0),
		m_size(0),
		m_deallocEnabled(false),
		m_valid(false),
		m_noValidDeviceCopy(true)
	{
		this->swap(c);
	}
	
	
	template <typename T>
	inline Vector<T>::Vector(std::initializer_list<T> l): m_capacity(l.size()), m_size(l.size()), m_deallocEnabled(true), m_valid(true), m_noValidDeviceCopy(true)
	{
		backend::allocateHostMemory<T>(m_data, m_capacity);
		
		int i = 0;
		for (const T& elem : l)
			m_data[i++] = elem;
	}
	
	
	/**!
	 * Used to construct vector on a raw data pointer passed to it as its payload data.
	 * Useful when creating the vector object with existing raw data pointer.
	 */
	template <typename T>
	inline Vector<T>::Vector(T * const ptr, size_type size, bool deallocEnabled): m_capacity(size), m_size (size), m_deallocEnabled(deallocEnabled), m_valid(true), m_noValidDeviceCopy(true)
	{
		if (m_size < 1)
			SKEPU_ERROR("The vector size should be positive.");
		
		if (!ptr)
		{
			SKEPU_ERROR("Error: The supplied pointer for initializing vector object is invalid");
			return;
		}
		
		m_data = ptr;
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	inline Vector<T>::Vector(size_type num, const T& val): m_capacity(num), m_size(num), m_deallocEnabled(true), m_valid(true), m_noValidDeviceCopy(true)
	{
	//	if (m_size < 1)
		//	SKEPU_ERROR("The vector size should be positive.");
		
		backend::allocateHostMemory<T>(m_data, m_capacity);
		
		std::fill(m_data, m_data + m_size, val);
	}
	
///////////////////////////////////////////////
// Constructors END
///////////////////////////////////////////////

///////////////////////////////////////////////
// Destructor START
///////////////////////////////////////////////
	
	/*!
	 *  Releases all allocations made on device.
	 */
	template <typename T>
	Vector<T>::~Vector()
	{
		releaseDeviceAllocations();
		
		if (m_data && m_deallocEnabled)
			backend::deallocateHostMemory<T>(m_data);
	}
	
///////////////////////////////////////////////
// Destructor END
///////////////////////////////////////////////


///////////////////////////////////////////////
// Public Helpers START
///////////////////////////////////////////////

	/*!
	 *  Updates the vector from its device allocations.
	 */
	template <typename T>
	inline void Vector<T>::updateHost(bool enable) const
	{
		if (!enable)
			return;
	
#ifdef SKEPU_OPENCL
		updateHost_CL();
#endif

#ifdef SKEPU_CUDA
   /*! the m_valid logic is only implemented for CUDA backend. The OpenCL still uses the old memory management mechanism */
		if(m_valid) // if already up to date then no need to check...
			return;
   
		updateHost_CU();
#endif

		m_valid = true;
	}

	/*!
	 *  Invalidates (mark copies data invalid) all device data that this vector has allocated.
	 */
	template <typename T>
	inline void Vector<T>::invalidateDeviceData(bool enable) const
	{
		if (!enable)
			return;
	
#ifdef SKEPU_OPENCL
		invalidateDeviceData_CL();
#endif

#ifdef SKEPU_CUDA
		if(m_noValidDeviceCopy)
			assert(m_valid);
   
		if(!m_noValidDeviceCopy)
		{
			invalidateDeviceData_CU();
			m_noValidDeviceCopy = true;
			m_valid = true;
		}
#endif
	}

	/*!
	 *  First updates the vector from its device allocations. Then invalidates (mark copies data invalid) the data allocated on devices.
	 */
	template <typename T>
	inline void Vector<T>::updateHostAndInvalidateDevice()
	{
		updateHost();
		invalidateDeviceData();
	}

	/*!
	 *  Removes the data copies allocated on devices.
	 */
	template <typename T>
	inline void Vector<T>::releaseDeviceAllocations()
	{
#ifdef SKEPU_OPENCL
		releaseDeviceAllocations_CL();
#endif

#ifdef SKEPU_CUDA
		m_valid = true;
   
		releaseDeviceAllocations_CU();
#endif
	}

	/*!
	 *  First updates the vector from its device allocations. Then removes the data copies allocated on devices.
	 */
	template <typename T>
	inline void Vector<T>::updateHostAndReleaseDeviceAllocations()
	{
		updateHost();
		releaseDeviceAllocations();
	}


///////////////////////////////////////////////
// Public Helpers END
///////////////////////////////////////////////


///////////////////////////////////////////////
// Operators START
///////////////////////////////////////////////

	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	const T& Vector<T>::operator[](const size_type index) const
	{
		updateHost();
//    updateHostAndInvalidateDevice();

		return m_data[index];
	}

#ifdef SKEPU_PRECOMPILED
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 *
	 *  Returns a proxy_elem instead of an ordinary element. The proxy_elem usually
	 *  behaves like an ordinary, but there might be exceptions.
	 */
	template <typename T>
	typename Vector<T>::proxy_elem Vector<T>::operator[](const size_type index)
	{
		return proxy_elem(*this, index);
	}
	
#else
	
	template<typename T>
	T& Vector<T>::operator[](const size_type index)
	{
		return this->m_data[index];
	}
	
#endif // SKEPU_PRECOMPILED
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	Vector<T>& Vector<T>::operator=(const Vector<T>& other)
	{
		if (*this == other)
			return *this;
		 
		updateHostAndReleaseDeviceAllocations();
		other.updateHost();
		
		if (m_capacity < other.m_size)
		{
			if (m_data)
				backend::deallocateHostMemory<T>(m_data);
			
			m_capacity = m_size = other.m_size;
			
			backend::allocateHostMemory<T>(m_data, m_capacity);
		}
		else
		{
			m_size = other.m_size;
		}
		
		std::copy(other.m_data, other.m_data + m_size, m_data);
		
		return *this;
	}

///////////////////////////////////////////////
// Operators END
///////////////////////////////////////////////


///////////////////////////////////////////////
// Regular interface functions START
///////////////////////////////////////////////
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	typename Vector<T>::iterator Vector<T>::begin()
	{
		return iterator(*this, &m_data[0]);
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	typename Vector<T>::iterator Vector<T>::end()
	{
		return iterator(*this, &m_data[m_size]);
	}
	
	
		/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	typename Vector<T>::const_iterator Vector<T>::begin() const
	{
		return const_iterator(*this, &m_data[0]);
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	typename Vector<T>::const_iterator Vector<T>::end() const
	{
		return const_iterator(*this, &m_data[m_size]);
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	void Vector<T>::resize(size_type num, T val)
	{
		updateHostAndReleaseDeviceAllocations();
		
		if (num <= m_size) // dont shrink the size, maybe good in some cases?
		{
			this->m_size = num;
			return;
		}
		
		reserve(num);
		std::fill(this->m_data + this->m_size, this->m_data + num, val);
		
		this->m_size = num;
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	bool Vector<T>::empty() const
	{
		return m_size == 0;
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	void Vector<T>::reserve(size_type size)
	{
		if (size <= this->m_capacity)
			return;
		
		updateHostAndReleaseDeviceAllocations();
		
		T* temp;
		
		backend::allocateHostMemory<T>(temp, size);
		std::copy(this->m_data, this->m_data + this->m_size, temp);
		backend::deallocateHostMemory<T>(m_data);
		
		this->m_data = temp;
		this->m_capacity = size;
		temp = 0;
	}
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	const T& Vector<T>::at(size_type loc) const
	{
		updateHost();
		return this->m_data[loc];
	}
	
#ifdef SKEPU_PRECOMPILED
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 *
	 *  Returns a \p proxy_elem instead of an ordinary element. The \p proxy_elem usually
	 *  behaves like an ordinary, but there might be exceptions.
	 */
	template <typename T>
	typename Vector<T>::proxy_elem Vector<T>::at(size_type loc)
	{
		return proxy_elem(*this, loc);
	}
	
#else
	
	template <typename T>
	T& Vector<T>::at(size_type loc)
	{
		return this->m_data[loc];
	}
	
#endif // SKEPU_PRECOMPILED
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	const T& Vector<T>::back() const
	{
		updateHost();
		return this->m_data[m_size-1];
	}
	
#ifdef SKEPU_PRECOMPILED
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 *
	 *  Returns a \p proxy_elem instead of an ordinary element. The \p proxy_elem usually
	 *  behaves like an ordinary, but there might be exceptions.
	 */
	template <typename T>
	typename Vector<T>::proxy_elem Vector<T>::back()
	{
		return proxy_elem(*this, m_size-1);
	}
	
#else
	
	template <typename T>
	T& Vector<T>::back()
	{
		return this->m_data[this->m_size-1];
	}
	
#endif // SKEPU_PRECOMPILED
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	const T& Vector<T>::front() const
	{
		updateHost();
		return m_data[0];
	}
	
#ifdef SKEPU_PRECOMPILED
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 *
	 *  Returns a \p proxy_elem instead of an ordinary element. The \p proxy_elem usually
	 *  behaves like an ordinary, but there might be exceptions.
	 */
	template <typename T>
	typename Vector<T>::proxy_elem Vector<T>::front()
	{
		return proxy_elem(*this, 0);
	}
	
#else
	
	template <typename T>
	T& Vector<T>::front()
	{
		return m_data[0];
	}
	
#endif // SKEPU_PRECOMPILED
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	void Vector<T>::assign(size_type num, const T& val)
	{
		releaseDeviceAllocations();
		
		reserve(num); // check if need some reallocation
		std::fill(m_data, m_data + num, val);
		m_size = num;
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	template<typename input_iterator>
	void Vector<T>::assign( input_iterator start, input_iterator end )
	{
		updateHostAndReleaseDeviceAllocations();
		
		size_type num = end-start;
		reserve(num); // check if need some reallocation
		
		for (size_type i = 0; i < num; i++, start++)
		{
			m_data[i]=*start;
		}
		
		m_size = num;
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	void Vector<T>::clear()
	{
		releaseDeviceAllocations();
		
		m_size=0;
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	typename Vector<T>::iterator Vector<T>::erase( typename Vector<T>::iterator loc )
	{
		updateHostAndReleaseDeviceAllocations();
		
		std::copy(loc+1, end(), loc);
		--m_size;
		return loc;
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	typename Vector<T>::iterator Vector<T>::erase( typename Vector<T>::iterator start, typename Vector<T>::iterator end )
	{
		updateHostAndReleaseDeviceAllocations();
		
		std::copy(end, iterator(*this, &m_data[m_size]), start);
		m_size -= (end-start);
		return start;
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	typename Vector<T>::iterator Vector<T>::insert( typename Vector<T>::iterator loc, const T& val )
	{
		updateHostAndReleaseDeviceAllocations();
		
		reserve(m_size+1);
		copy(loc, end(), loc+1);
		++m_size;
		*loc = val;
		
		return loc;
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	void Vector<T>::insert( typename Vector<T>::iterator loc, size_type num, const T& val )
	{
		updateHostAndReleaseDeviceAllocations();
		
		reserve(m_size+num);
		
		copy(loc, end(), loc+num);
		m_size += num;
		
		for(size_type i=0; i<num; i++)
		{
			*loc = val;
			++loc;
		}
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	void Vector<T>::pop_back()
	{
		updateHostAndReleaseDeviceAllocations();
		--m_size;
	}

	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	void Vector<T>::push_back(const T& val)
	{
		updateHostAndReleaseDeviceAllocations();
		
		if (m_size >= m_capacity)
			reserve(m_capacity + 5);
		
		m_data[m_size++] = val;
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	void Vector<T>::swap(Vector<T>& from)
	{
		updateHostAndReleaseDeviceAllocations();
		from.updateHostAndReleaseDeviceAllocations();
		
		std::swap(m_data, from.m_data);
		std::swap(m_size, from.m_size);
		std::swap(m_capacity, from.m_capacity);
	}

///////////////////////////////////////////////
// Regular interface functions END
///////////////////////////////////////////////


///////////////////////////////////////////////
// Additions to interface START
///////////////////////////////////////////////

	/*!
	 *  Flushes the vector, synchronizing it with the device then release all device allocations.
	 */
	template <typename T>
	void Vector<T>::flush()
	{
#ifdef SKEPU_OPENCL
		flush_CL();
#endif

#ifdef SKEPU_CUDA
		flush_CU();
#endif
	}

///////////////////////////////////////////////
// Additions to interface END
///////////////////////////////////////////////


///////////////////////////////////////////////
// Comparison operators START
///////////////////////////////////////////////

	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	bool Vector<T>::operator==(const Vector<T>& c1)
	{
		c1.updateHost();
		updateHost();
		
		if (m_size != c1.m_size)
			return false;
		
		if (m_data == c1.m_data)
			return true;
		
		for (size_type i = 0; i < m_size; i++)
		{
			if (m_data[i] != c1.m_data[i])
				return false;
		}
		return true;
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	bool Vector<T>::operator!=(const Vector<T>& c1)
	{
		c1.updateHost();
		updateHost();
		
		if (m_size != c1.m_size)
			return true;
		
		if (m_data == c1.m_data && m_size == c1.m_size)
			return false;
		
		for (size_type i = 0; i < m_size; i++)
		{
			if (m_data[i] != c1.m_data[i])
				return true;
		}
		return false;
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	bool Vector<T>::operator<(const Vector<T>& c1)
	{
		c1.updateHost();
		updateHost();
		
		size_type t_size = (c1.size() < size()) ? c1.size() : size();
		
		for (size_type i = 0; i < t_size; ++i)
		{
			if (m_data[i] >= c1.m_data[i])
				return false;
		}
		return true;
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	bool Vector<T>::operator>(const Vector<T>& c1)
	{
		c1.updateHost();
		updateHost();
		
		size_type t_size = (c1.size() < size()) ? c1.size() : size();
		
		for (size_type i = 0; i < t_size; ++i)
		{
			if (m_data[i] <= c1.m_data[i])
				return false;
		}
		return true;
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	bool Vector<T>::operator<=(const Vector<T>& c1)
	{
		c1.updateHost();
		updateHost();
		
		size_type t_size = (c1.size() < size()) ? c1.size() : size();
		
		for (size_type i = 0; i < t_size; ++i)
		{
			if (m_data[i] > c1.m_data[i])
				return false;
		}
		return true;
	}
	
	
	/*!
	 *  Please refer to the documentation of \p std::vector.
	 */
	template <typename T>
	bool Vector<T>::operator>=(const Vector<T>& c1)
	{
		c1.updateHost();
		updateHost();
		
		size_type t_size = (c1.size() < size()) ? c1.size() : size();
		
		for (size_type i = 0; i < t_size; ++i)
		{
			if (m_data[i] < c1.m_data[i])
				return false;
		}
		return true;
	}

///////////////////////////////////////////////
// Comparison operators END
///////////////////////////////////////////////

}


#include "vector_iterator.inl"

#ifdef SKEPU_PRECOMPILED

#include "vector_proxy.inl"
#include "vector_cl.inl"
#include "vector_cu.inl"

#endif // SKEPU_PRECOMPILED
