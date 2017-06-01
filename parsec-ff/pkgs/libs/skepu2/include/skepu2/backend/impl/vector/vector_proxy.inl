namespace skepu2
{

/*!
 *  \class Vector::proxy_elem
 *  \author Johan Enmyren, Usman Dastgeer
 *  \version 0.7
 *
 *  \brief A proxy class representing one element of Vector.
 *
 *  A member class of skepu::Vector that represents a proxy element which is returned instead
 *  of a regular element in some functions. Its purpose is to be able to distinguish between
 *  a read and a write from the vector. In most cases it is compatible with an ordinary element
 *  but there are cases where it is not.
 *
 *  \see Scott Meyers "More effective C++" §30.
 */
template <typename T>
class Vector<T>::proxy_elem
{

public: //-- Constructors & Destructor --//

   proxy_elem(Vector<T>& m_parent, const size_type index);


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

   Vector<T>& m_parent;
   const size_type m_index;

   void updateAndInvalidate(Vector<T>& target);
   void update(Vector<T>& target) const;

};

template <typename T>
Vector<T>::proxy_elem::proxy_elem(Vector<T>& parent, const size_type index) : m_parent(parent), m_index(index) {}

template <typename T>
inline void Vector<T>::proxy_elem::updateAndInvalidate(Vector<T>& target)
{
   target.updateHost();
   target.invalidateDeviceData();
}

template <typename T>
inline void Vector<T>::proxy_elem::update(Vector<T>& target) const
{
   target.updateHost();
}

//This is where values are being read
template <typename T>
Vector<T>::proxy_elem::operator T&() const
{
   update(m_parent);

   return m_parent.m_data[m_index];
}

//This is where values are being written
template <typename T>
typename Vector<T>::proxy_elem& Vector<T>::proxy_elem::operator=(const proxy_elem& rhs)
{
   update(rhs.m_parent);
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] = rhs.m_parent.m_data[rhs.m_index];
   return *this;
}

template <typename T>
typename Vector<T>::proxy_elem& Vector<T>::proxy_elem::operator=(const T& rhs)
{
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] = rhs;

   return *this;
}

//Other, might change vector or might not, invalidate to be sure
template <typename T>
T* Vector<T>::proxy_elem::operator&()
{
   updateAndInvalidate(m_parent);

   return &(m_parent.m_data[m_index]);
}

template <typename T>
const typename Vector<T>::proxy_elem& Vector<T>::proxy_elem::operator++()
{
   updateAndInvalidate(m_parent);

   ++m_parent.m_data[m_index];
   return *this;
}

template <typename T>
typename Vector<T>::proxy_elem Vector<T>::proxy_elem::operator++(int)
{
   updateAndInvalidate(m_parent);

   proxy_elem temp(*this);
   m_parent.m_data[m_index]++;
   return temp;
}

template <typename T>
const typename Vector<T>::proxy_elem& Vector<T>::proxy_elem::operator--()
{
   updateAndInvalidate(m_parent);

   --m_parent.m_data[m_index];
   return *this;
}

template <typename T>
typename Vector<T>::proxy_elem Vector<T>::proxy_elem::operator--(int)
{
   updateAndInvalidate(m_parent);

   proxy_elem temp(*this);
   m_parent.m_data[m_index]--;
   return temp;
}

template <typename T>
const typename Vector<T>::proxy_elem& Vector<T>::proxy_elem::operator+=(const T& rhs)
{
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] += rhs;
   return *this;
}

template <typename T>
const typename Vector<T>::proxy_elem& Vector<T>::proxy_elem::operator+=(const proxy_elem& rhs)
{
   update(rhs.m_parent);
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] += rhs.m_parent.m_data[rhs.m_index];
   return *this;
}

template <typename T>
const typename Vector<T>::proxy_elem& Vector<T>::proxy_elem::operator-=(const T& rhs)
{
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] -= rhs;
   return *this;
}

template <typename T>
const typename Vector<T>::proxy_elem& Vector<T>::proxy_elem::operator-=(const proxy_elem& rhs)
{
   update(rhs.m_parent);
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] -= rhs.m_parent.m_data[rhs.m_index];
   return *this;
}

template <typename T>
const typename Vector<T>::proxy_elem& Vector<T>::proxy_elem::operator*=(const T& rhs)
{
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] *= rhs;
   return *this;
}

template <typename T>
const typename Vector<T>::proxy_elem& Vector<T>::proxy_elem::operator*=(const proxy_elem& rhs)
{
   update(rhs.m_parent);
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] *= rhs.m_parent.m_data[rhs.m_index];
   return *this;
}

template <typename T>
const typename Vector<T>::proxy_elem& Vector<T>::proxy_elem::operator/=(const T& rhs)
{
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] /= rhs;
   return *this;
}

template <typename T>
const typename Vector<T>::proxy_elem& Vector<T>::proxy_elem::operator/=(const proxy_elem& rhs)
{
   update(rhs.m_parent);
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] /= rhs.m_parent.m_data[rhs.m_index];
   return *this;
}

template <typename T>
const typename Vector<T>::proxy_elem& Vector<T>::proxy_elem::operator<<=(const T& rhs)
{
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] <<= rhs;
   return *this;
}

template <typename T>
const typename Vector<T>::proxy_elem& Vector<T>::proxy_elem::operator<<=(const proxy_elem& rhs)
{
   update(rhs.m_parent);
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] <<= rhs.m_parent.m_data[rhs.m_index];
   return *this;
}

template <typename T>
const typename Vector<T>::proxy_elem& Vector<T>::proxy_elem::operator>>=(const T& rhs)
{
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] >>= rhs;
   return *this;
}

template <typename T>
const typename Vector<T>::proxy_elem& Vector<T>::proxy_elem::operator>>=(const proxy_elem& rhs)
{
   update(rhs.m_parent);
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] >>= rhs.m_parent.m_data[rhs.m_index];
   return *this;
}

template <typename T>
const typename Vector<T>::proxy_elem& Vector<T>::proxy_elem::operator&=(const T& rhs)
{
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] &= rhs;
   return *this;
}

template <typename T>
const typename Vector<T>::proxy_elem& Vector<T>::proxy_elem::operator&=(const proxy_elem& rhs)
{
   update(rhs.m_parent);
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] &= rhs.m_parent.m_data[rhs.m_index];
   return *this;
}

template <typename T>
const typename Vector<T>::proxy_elem& Vector<T>::proxy_elem::operator|=(const T& rhs)
{
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] |= rhs;
   return *this;
}

template <typename T>
const typename Vector<T>::proxy_elem& Vector<T>::proxy_elem::operator|=(const proxy_elem& rhs)
{
   update(rhs.m_parent);
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] |= rhs.m_parent.m_data[rhs.m_index];
   return *this;
}

template <typename T>
const typename Vector<T>::proxy_elem& Vector<T>::proxy_elem::operator^=(const T& rhs)
{
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] ^= rhs;
   return *this;
}

template <typename T>
const typename Vector<T>::proxy_elem& Vector<T>::proxy_elem::operator^=(const proxy_elem& rhs)
{
   update(rhs.m_parent);
   updateAndInvalidate(m_parent);

   m_parent.m_data[m_index] ^= rhs.m_parent.m_data[rhs.m_index];
   return *this;
}

}

