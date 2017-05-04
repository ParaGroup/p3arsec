/*! \file matrix_iterator.inl
 *  \brief Contains the definitions of Matrix::iterator class.
 */

namespace skepu2
{
	template<typename T>
	Index2D MatrixIterator<T>::getIndex() const
	{
		size_t index = *this - m_parent->begin();
		size_t rows = index / m_parent->total_cols();
		size_t cols = index % m_parent->total_cols();
		return Index2D{rows, cols};
	}
	
	
	template<typename T>
	MatrixIterator<T>::MatrixIterator(parent_type *parent, const iterator_type std_iterator) : m_parent(parent), m_std_iterator(std_iterator)
	{}
	
	
	template<typename T>
	typename MatrixIterator<T>::parent_type& MatrixIterator<T>::getParent() const
	{
		return (*m_parent);
	}
	
	
	template<typename T>
	MatrixIterator<T>& MatrixIterator<T>::begin()
	{
		return *this;
	}
	
	
	template<typename T>
	size_t MatrixIterator<T>::size()
	{
		return this->m_parent->end() - *this;
	}
	
	
	template<typename T>
	T* MatrixIterator<T>::getAddress() const
	{
		return &(*m_std_iterator);
	}
	
	
// Does not care about device data, use with care...
	template<typename T>
	T& MatrixIterator<T>::operator()(const ssize_t index)
	{
		return m_std_iterator[index];
	}

	template<typename T>
	T& MatrixIterator<T>::operator()(const ssize_t row, const ssize_t col)
	{
		m_parent->updateHost();
		m_parent->invalidateDeviceData();
		return m_std_iterator[(row*getParent().total_cols() + col)];
	}
	
	
	template<typename T>
	const T& MatrixIterator<T>::operator()(const ssize_t row, const ssize_t col) const
	{
		m_parent->updateHost();
		return m_std_iterator[(row*getParent().total_cols() + col)];
	}
	
	
	template<typename T>
	MatrixIterator<T>& MatrixIterator<T>::stride_row(const ssize_t stride)
	{
		return m_std_iterator += (stride * getParent().total_cols());
	}
	
	
	template<typename T>
	T& MatrixIterator<T>::operator[](const ssize_t index)
	{
		m_parent->updateHost();
		m_parent->invalidateDeviceData();
		return m_std_iterator[index];
	}
	
	
	template<typename T>
	const T& MatrixIterator<T>::operator[](const ssize_t index) const
	{
		m_parent->updateHost();
		return m_std_iterator[index];
	}
	
	
/*	template<typename T>
	MatrixIterator<T>::operator const_iterator() const
	{
		m_parent->updateHost();
		return static_cast< const_iterator > (m_std_iterator);
	}*/
	
	
/*	template<typename T>
	MatrixIterator<T>::operator iterator_type() const
	{
		m_parent->updateHost();
		m_parent->invalidateDeviceData();
		return m_std_iterator;
	}*/
	
	
	template<typename T>
	bool MatrixIterator<T>::operator==(const iterator& i)
	{
		return (m_std_iterator == i.m_std_iterator);
	}
	
	
	template<typename T>
	bool MatrixIterator<T>::operator!=(const iterator& i)
	{
		return (m_std_iterator != i.m_std_iterator);
	}
	
	
	template<typename T>
	bool MatrixIterator<T>::operator<(const iterator& i)
	{
		return (m_std_iterator < i.m_std_iterator);
	}
	
	
	template<typename T>
	bool MatrixIterator<T>::operator>(const iterator& i)
	{
		return (m_std_iterator > i.m_std_iterator);
	}
	
	
	template<typename T>
	bool MatrixIterator<T>::operator<=(const iterator& i)
	{
		return (m_std_iterator <= i.m_std_iterator);
	}
	
	
	template<typename T>
	bool MatrixIterator<T>::operator>=(const iterator& i)
	{
		return (m_std_iterator >= i.m_std_iterator);
	}
	
	
	template<typename T>
	const MatrixIterator<T>& MatrixIterator<T>::operator++() //Prefix
	{
		++m_std_iterator;
		return *this;
	}
	
	
	template<typename T>
	MatrixIterator<T> MatrixIterator<T>::operator++(int) //Postfix
	{
		iterator temp(*this);
		++m_std_iterator;
		return temp;
	}
	
	
	template<typename T>
	const MatrixIterator<T>& MatrixIterator<T>::operator--() //Prefix
	{
		--m_std_iterator;
		return *this;
	}
	
	
	template<typename T>
	MatrixIterator<T> MatrixIterator<T>::operator--(int) //Postfix
	{
		iterator temp(*this);
		--m_std_iterator;
		return temp;
	}
	
	
	template<typename T>
	const MatrixIterator<T>& MatrixIterator<T>::operator+=(const ssize_t i)
	{
		m_std_iterator += i;
		return *this;
	}
	
	
	template<typename T>
	const MatrixIterator<T>& MatrixIterator<T>::operator-=(const ssize_t i)
	{
		m_std_iterator -= i;
		return *this;
	}
	
	
	template<typename T>
	MatrixIterator<T> MatrixIterator<T>::operator-(const ssize_t i) const
	{
		iterator temp(*this);
		temp -= i;
		return temp;
	}
	
	
	template<typename T>
	MatrixIterator<T> MatrixIterator<T>::operator+(const ssize_t i) const
	{
		iterator temp(*this);
		temp += i;
		return temp;
	}
	
	
	template<typename T>
	typename MatrixIterator<T>::parent_type::difference_type MatrixIterator<T>::operator-(const iterator& i) const
	{
		return m_std_iterator - i.m_std_iterator;
	}
	
	
	template<typename T>
	T& MatrixIterator<T>::operator*()
	{
		m_parent->updateHost();
		m_parent->invalidateDeviceData();
		return *m_std_iterator;
	}
	
	
	template<typename T>
	const T& MatrixIterator<T>::operator*() const
	{
		m_parent->updateHost();
		return *m_std_iterator;
	}
	
	
	template<typename T>
	const T& MatrixIterator<T>::operator-> () const
	{
		m_parent->updateHost();
		return *m_std_iterator;
	}
	
	
	template<typename T>
	T& MatrixIterator<T>::operator-> ()
	{
		m_parent->updateHost();
		m_parent->invalidateDeviceData();
		return *m_std_iterator;
	}
	
}
