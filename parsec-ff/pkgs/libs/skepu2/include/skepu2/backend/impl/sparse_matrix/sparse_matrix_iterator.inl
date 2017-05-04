/*! \file sparse_matrix_iterator.inl
 *  \brief Contains the definitions of the SparseMatrix::iterator class.
 */

namespace skepu2
{
/*
template <typename T>
index2D SparseMatrix<T>::iterator::getIndex() const
{
	size_t index = *this - m_parent->begin();
	size_t rows = index / m_parent->total_cols();
	size_t cols = index % m_parent->total_cols();
	return index2D{rows, cols};
}*/

template <typename T>
SparseMatrix<T>::iterator::iterator(SparseMatrix<T>* parent, T *start, T *end) : m_parent(parent), m_start(start), m_end(end)
{
   SKEPU_ASSERT(m_start);
   SKEPU_ASSERT(m_end);
   SKEPU_ASSERT((m_end-m_start)>=0);
   m_size = (m_end-m_start);
}

template <typename T>
T* SparseMatrix<T>::iterator::getAddress() const
{
   return m_start;
}

// Does not care about device data, use with care...
template <typename T>
const T& SparseMatrix<T>::iterator::operator()(const ssize_t index) const
{
   return m_start[index];
}

template <typename T>
T& SparseMatrix<T>::iterator::operator()(const ssize_t index)
{
   return m_start[index];
}


template <typename T>
const T& SparseMatrix<T>::iterator::operator[](const ssize_t index) const
{
#ifdef SKEPU_OPENCL
   m_parent->updateHost_CL();
#endif

#ifdef SKEPU_CUDA
   m_parent->updateHost_CU();
#endif

   return m_start[index];
}


template <typename T>
T& SparseMatrix<T>::iterator::operator[](const ssize_t index)
{
#ifdef SKEPU_OPENCL
   m_parent->updateHost_CL();
   m_parent->invalidateDeviceData_CL();
#endif

#ifdef SKEPU_CUDA
   m_parent->updateHost_CU();
   m_parent->invalidateDeviceData_CU();
#endif

   return m_start[index];
}

/*
template <typename T>
const typename SparseMatrix<T>::iterator& SparseMatrix<T>::iterator::operator++() //Prefix
{
   ++m_start;
   return *this;
}

template <typename T>
typename SparseMatrix<T>::difference_type SparseMatrix<T>::iterator::operator-(const iterator& i) const
{
   return m_start - i.m_start;
}

template <typename T>
const T& SparseMatrix<T>::iterator::operator*() const
{
#ifdef SKEPU_OPENCL
	m_parent->updateHost_CL();
#endif

#ifdef SKEPU_CUDA
	m_parent->updateHost_CU();
#endif

   return m_start[0];
}*/

}

