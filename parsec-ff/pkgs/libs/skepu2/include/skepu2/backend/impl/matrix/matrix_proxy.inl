/*! \file matrix_proxy.inl
 *  \brief Contains the definitions of the Matrix::proxy_elem class.
 */


namespace skepu2
{

/*!
 *  \brief A proxy class representing one element of Matrix.
 *
 *  A member class of skepu::Matrix that represents a proxy element which is returned instead
 *  of a regluar element in some functions. It is very similar to \p skepu::Vector::proxy_elem.
 *  Its purpose is to be able to distinguish between
 *  a read and a write from the vector. In most cases it is compatible with an ordinary element
 *  but there are cases where it is not.
 *
 *  \see Scott Meyers "More effective C++".
 */
template <typename T>
class Matrix<T>::proxy_elem
{

public: //-- Constructors & Destructor --//

   proxy_elem(Matrix<T>& m_parent, const size_type index);
   proxy_elem(Matrix<T>& m_parent, const size_type rowIndex, const size_type colIndex);


public: //-- Operators --//

   //Lvalue uses
   proxy_elem& operator=(const proxy_elem& rhs);
   proxy_elem& operator=(const T& rhs);

   //Rvalue uses
   operator T&() const;

   //Other
   T* operator&();

   const proxy_elem& operator++();
   proxy_elem operator++(int);
   const proxy_elem& operator--();
   proxy_elem operator--(int);
   const proxy_elem& operator+=(const T& rhs);
   const proxy_elem& operator+=(const proxy_elem& rhs);
   const proxy_elem& operator-=(const T& rhs);
   const proxy_elem& operator-=(const proxy_elem& rhs);
   const proxy_elem& operator*=(const T& rhs);
   const proxy_elem& operator*=(const proxy_elem& rhs);
   const proxy_elem& operator/=(const T& rhs);
   const proxy_elem& operator/=(const proxy_elem& rhs);
   const proxy_elem& operator%=(const T& rhs);
   const proxy_elem& operator%=(const proxy_elem& rhs);
   const proxy_elem& operator<<=(const T& rhs);
   const proxy_elem& operator<<=(const proxy_elem& rhs);
   const proxy_elem& operator>>=(const T& rhs);
   const proxy_elem& operator>>=(const proxy_elem& rhs);
   const proxy_elem& operator&=(const T& rhs);
   const proxy_elem& operator&=(const proxy_elem& rhs);
   const proxy_elem& operator|=(const T& rhs);
   const proxy_elem& operator|=(const proxy_elem& rhs);
   const proxy_elem& operator^=(const T& rhs);
   const proxy_elem& operator^=(const proxy_elem& rhs);


private: //-- Data --//

   Matrix<T>& m_parent;
   size_type m_index;

   void updateAndInvalidate(Matrix<T>& target);
   void update(Matrix<T>& target) const;

};

template <typename T>
Matrix<T>::proxy_elem::proxy_elem(Matrix<T>& parent, const size_type index) : m_parent(parent), m_index(index) {}

template <typename T>
Matrix<T>::proxy_elem::proxy_elem(Matrix<T>& parent, const size_type rowIndex, const size_type colIndex) : m_parent(parent)
{
   m_index = (rowIndex*parent.total_cols()) + colIndex;
}

template <typename T>
inline void Matrix<T>::proxy_elem::updateAndInvalidate(Matrix<T>& target)
{
   target.updateHost();
   target.invalidateDeviceData();
}

template <typename T>
inline void Matrix<T>::proxy_elem::update(Matrix<T>& target) const
{
   target.updateHost();
}

//This is where values are being read
template <typename T>
Matrix<T>::proxy_elem::operator T&() const
{
   update(m_parent);

   return m_parent.m_data[m_index];
}

//This is where values are being written
template <typename T>
typename Matrix<T>::proxy_elem& Matrix<T>::proxy_elem::operator=(const proxy_elem& rhs)
{
   update(rhs.m_parent);
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] = rhs.m_parent.m_data[rhs.m_index];
   return *this;
}

template <typename T>
typename Matrix<T>::proxy_elem& Matrix<T>::proxy_elem::operator=(const T& rhs)
{
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] = rhs;

   return *this;
}

//Other, might change vector or might not, invalidate to be sure
template <typename T>
T* Matrix<T>::proxy_elem::operator&()
{
   updateAndInvalidate(m_parent);

   return &(m_parent.m_data[m_index]);
}

template <typename T>
const typename Matrix<T>::proxy_elem& Matrix<T>::proxy_elem::operator++()
{
   updateAndInvalidate(m_parent);

   ++m_parent.m_data[m_index];
   return *this;
}

template <typename T>
typename Matrix<T>::proxy_elem Matrix<T>::proxy_elem::operator++(int)
{
   updateAndInvalidate(m_parent);

   proxy_elem temp(*this);
   m_parent.m_data[m_index]++;
   return temp;
}

template <typename T>
const typename Matrix<T>::proxy_elem& Matrix<T>::proxy_elem::operator--()
{
   updateAndInvalidate(m_parent);

   --m_parent.m_data[m_index];
   return *this;
}

template <typename T>
typename Matrix<T>::proxy_elem Matrix<T>::proxy_elem::operator--(int)
{
   updateAndInvalidate(m_parent);

   proxy_elem temp(*this);
   m_parent.m_data[m_index]--;
   return temp;
}

template <typename T>
const typename Matrix<T>::proxy_elem& Matrix<T>::proxy_elem::operator+=(const T& rhs)
{
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] += rhs;
   return *this;
}

template <typename T>
const typename Matrix<T>::proxy_elem& Matrix<T>::proxy_elem::operator+=(const proxy_elem& rhs)
{
   update(rhs.m_parent);
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] += rhs.m_parent.m_data[rhs.m_index];
   return *this;
}

template <typename T>
const typename Matrix<T>::proxy_elem& Matrix<T>::proxy_elem::operator-=(const T& rhs)
{
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] -= rhs;
   return *this;
}

template <typename T>
const typename Matrix<T>::proxy_elem& Matrix<T>::proxy_elem::operator-=(const proxy_elem& rhs)
{
   update(rhs.m_parent);
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] -= rhs.m_parent.m_data[rhs.m_index];
   return *this;
}

template <typename T>
const typename Matrix<T>::proxy_elem& Matrix<T>::proxy_elem::operator*=(const T& rhs)
{
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] *= rhs;
   return *this;
}

template <typename T>
const typename Matrix<T>::proxy_elem& Matrix<T>::proxy_elem::operator*=(const proxy_elem& rhs)
{
   update(rhs.m_parent);
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] *= rhs.m_parent.m_data[rhs.m_index];
   return *this;
}

template <typename T>
const typename Matrix<T>::proxy_elem& Matrix<T>::proxy_elem::operator/=(const T& rhs)
{
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] /= rhs;
   return *this;
}

template <typename T>
const typename Matrix<T>::proxy_elem& Matrix<T>::proxy_elem::operator/=(const proxy_elem& rhs)
{
   update(rhs.m_parent);
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] /= rhs.m_parent.m_data[rhs.m_index];
   return *this;
}

template <typename T>
const typename Matrix<T>::proxy_elem& Matrix<T>::proxy_elem::operator<<=(const T& rhs)
{
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] <<= rhs;
   return *this;
}

template <typename T>
const typename Matrix<T>::proxy_elem& Matrix<T>::proxy_elem::operator<<=(const proxy_elem& rhs)
{
   update(rhs.m_parent);
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] <<= rhs.m_parent.m_data[rhs.m_index];
   return *this;
}

template <typename T>
const typename Matrix<T>::proxy_elem& Matrix<T>::proxy_elem::operator>>=(const T& rhs)
{
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] >>= rhs;
   return *this;
}

template <typename T>
const typename Matrix<T>::proxy_elem& Matrix<T>::proxy_elem::operator>>=(const proxy_elem& rhs)
{
   update(rhs.m_parent);
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] >>= rhs.m_parent.m_data[rhs.m_index];
   return *this;
}

template <typename T>
const typename Matrix<T>::proxy_elem& Matrix<T>::proxy_elem::operator&=(const T& rhs)
{
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] &= rhs;
   return *this;
}

template <typename T>
const typename Matrix<T>::proxy_elem& Matrix<T>::proxy_elem::operator&=(const proxy_elem& rhs)
{
   update(rhs.m_parent);
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] &= rhs.m_parent.m_data[rhs.m_index];
   return *this;
}

template <typename T>
const typename Matrix<T>::proxy_elem& Matrix<T>::proxy_elem::operator|=(const T& rhs)
{
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] |= rhs;
   return *this;
}

template <typename T>
const typename Matrix<T>::proxy_elem& Matrix<T>::proxy_elem::operator|=(const proxy_elem& rhs)
{
   update(rhs.m_parent);
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] |= rhs.m_parent.m_data[rhs.m_index];
   return *this;
}

template <typename T>
const typename Matrix<T>::proxy_elem& Matrix<T>::proxy_elem::operator^=(const T& rhs)
{
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] ^= rhs;
   return *this;
}

template <typename T>
const typename Matrix<T>::proxy_elem& Matrix<T>::proxy_elem::operator^=(const proxy_elem& rhs)
{
   update(rhs.m_parent);
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] ^= rhs.m_parent.m_data[rhs.m_index];
   return *this;
}

}

