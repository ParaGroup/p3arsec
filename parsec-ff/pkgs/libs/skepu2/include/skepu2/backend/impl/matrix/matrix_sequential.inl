/*! \file matrix.inl
 *  \brief Contains the definitions of non-backend specific member functions for the Matrix container.
 */

namespace skepu2
{

template<typename T>
void Matrix<T>::setValidFlag(bool val)
{
   m_valid = val;
}

/*!
* Get array representation
*/
template<typename T>
T* Matrix<T>::GetArrayRep()
{
   return &m_data[0];
}


/*!
*  \brief Randomizes the Matrix.
*
*  Sets each element of the Matrix to a random number between \p min and \p max.
*  The numbers are generated as \p integers but are cast to the type of the matrix.
*
*  \param min The smallest number an element can become.
*  \param max The largest number an element can become.
*/
template<typename T>
void Matrix<T>::randomize(int min, int max)
{
   invalidateDeviceData();

   for(typename Matrix<T>::size_type i = 0; i < size(); i++)
   {
      m_data.at(i) = (T)( rand() % (int)(max-min+1) + min);
      //            m_data.at(i) = min + (T)rand()/((T)RAND_MAX/(max-min));
   }
}

/*!
*  \brief Saves content of Matrix to a file.
*
*  Outputs the matrix as text on one line with space between elements to the specified file.
*  Mainly for testing purposes.
*
*  \param filename Name of file to save to.
*/
template<typename T>
void Matrix<T>::save(const std::string& filename)
{
   updateHost();

   std::ofstream file(filename.c_str());

   if (file.is_open())
   {
      for(size_type i = 0; i < m_data.size(); ++i)
      {
         file<<m_data.at(i) <<" ";
      }
      file.close();
   }
   else
   {
      std::cout<<"Unable to open file\n";
   }
}

/*!
*  \brief Loads the Matrix from a file.
*
*  Reads a variable number of elements from a file. In the file, all elemets should be in ASCII
*  on one line with whitespace between each element. Mainly for testing purposes.
*
*  \param filename Name of file to save to.
*  \param rowWidth The width of a row. All rows get same amount of width.
*  \param numRows The number of rows to be loaded. Default value 0 means all rows.
*/
template<typename T>
void Matrix<T>::load(const std::string& filename, size_type rowWidth, size_type numRows)
{
   invalidateDeviceData();

   std::ifstream file(filename.c_str());

   if (file.is_open())
   {
      std::string line;
      getline (file,line);
      std::istringstream ss(line);
      T num;
      clear();

      //Load all elements
      if(numRows == 0)
      {
         while(ss >> num)
         {
            push_back(num);
         }
      }
      // Load only numElements elements
      else
      {
         for(size_type i = 0; i < (numRows*rowWidth); ++i)
         {
            ss >> num;
            push_back(num);
         }
      }

      m_cols = rowWidth;
      m_rows = (size()/rowWidth);

      file.close();
   }
   else
   {
      std::cout<<"Unable to open file\n";
   }
}


/*!
* Private helper to swap any two elements of same type
*/
template<typename T>
template<typename Type>
void Matrix<T>::item_swap(Type &t1, Type &t2)
{
   Type temp= t1;
   t1=t2;
   t2=temp;
}

/*!
 *  Constructor, used to allocate memory ($_rows * _cols$).
 * \param _rows Number of rows in the matrix.
 * \param _cols Number of columns in the matrix.
 */
template<typename T>
Matrix<T>::Matrix(typename Matrix<T>::size_type _rows, typename Matrix<T>::size_type _cols)
: m_rows(_rows), m_cols(_cols), m_data(_rows * _cols), m_dataChanged(false), m_transpose_matrix(0), m_noValidDeviceCopy(true), m_valid(true)
{}

/*!
 *  Constructor, used to allocate memory ($_rows * _cols$). With a value ot initialize all elements.
 * \param _rows Number of rows in the matrix.
 * \param _cols Number of columns in the matrix.
 * \param val A value to initialize all elements.
 */
template<typename T>
Matrix<T>::Matrix(typename Matrix<T>::size_type _rows, typename Matrix<T>::size_type _cols, const T& val)
: m_rows(_rows), m_cols(_cols),m_data(m_rows * m_cols, val), m_dataChanged(false), m_transpose_matrix(0), m_noValidDeviceCopy(true), m_valid(true)
{}

/*!
 *  Constructor, used to allocate memory ($_rows * _cols$) with a vector to initialize all elements. 
 *  The size of the vector must be the same as _rows * _cols.
 * \param _rows Number of rows in the matrix.
 * \param _cols Number of columns in the matrix.
 * \param val A value to initialize all elements.
 */
template<typename T>
Matrix<T>::Matrix(typename Matrix<T>::size_type _rows, typename Matrix<T>::size_type _cols, const std::vector<T>& vals): 
	m_rows(_rows), 
	m_cols(_cols),
	m_data(vals), 
	m_dataChanged(false), 
	m_transpose_matrix(0), 
	m_noValidDeviceCopy(true), 
	m_valid(true)
{
	assert(m_data.size() == _rows * _cols);
}


/*!
 *  Constructor, initializes elements with data moved from argument vector. 
 *  The size of the vector must be the same as _rows * _cols.
 * \param _rows Number of rows in the matrix.
 * \param _cols Number of columns in the matrix.
 * \param val A value to initialize all elements.
 */
template<typename T>
Matrix<T>::Matrix(typename Matrix<T>::size_type _rows, typename Matrix<T>::size_type _cols, std::vector<T>&& vals): 
	m_rows(_rows), 
	m_cols(_cols),
	m_data(std::move(vals)), 
	m_dataChanged(false), 
	m_transpose_matrix(0), 
	m_noValidDeviceCopy(true), 
	m_valid(true)
{
	assert(m_data.size() == _rows * _cols);
}


/*!
 *  Copy Constructor, used to assign copy of another matrix.
 * \param copy Matrix that is being assigned.
 *
 * Update the matrix before assigning it to assign latest copy.
 */
template<typename T>
Matrix<T>::Matrix(const Matrix<T>& copy): m_noValidDeviceCopy(true), m_valid(true)
{
   copy.updateHost();
   this->m_rows = copy.m_rows;
   this->m_cols = copy.m_cols;
   this->m_data= copy.m_data;
   this->m_transpose_matrix = copy.m_transpose_matrix;
   this->m_dataChanged = copy.m_dataChanged;
}

template<typename T>
Matrix<T>::Matrix()
: m_rows(0), m_cols(0),m_data(), m_dataChanged(false), m_transpose_matrix(0), m_noValidDeviceCopy(true), m_valid(true)
{}


///////////////////////////////////////////////
// Operators START
///////////////////////////////////////////////




/*!
 *  copy matrix,,, copy row and column count as well along with data
 */
template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other)
{
   if(*this == other)
      return *this;
   
   other.updateHost();
   invalidateDeviceData();

   m_data = other.m_data;
   m_rows = other.m_rows;
   m_cols = other.m_cols;
   return *this;
}


/*!
 *  resize matrix,,, invalidates all copies before resizing.
 */
template <typename T>
void Matrix<T>::resize(size_type _rows, size_type _cols, T val)
{
   if (_rows == m_rows && _cols == m_cols)
   {
      return;
   }

   updateHostAndInvalidateDevice();

   typename Matrix<T>::container_type m( _rows*_cols, val);
   typename Matrix<T>::size_type colSize = std::min(m_cols,_cols) * sizeof(T);
   typename Matrix<T>::size_type minRow = std::min(m_rows,_rows);

   for (size_type r=0; r < _rows; r++)
   {
      for(size_type c=0; c < _cols; c++)
      {
         if(r < m_rows && c < m_cols)
            m[(r * _cols + c)]= m_data[(r * m_cols + c)];
         else
            m[(r * _cols + c)]= val;
      }
   }

   m_data=m;
   m_rows = _rows;
   m_cols = _cols;

}

/*!
 *  Add \p rhs matrix operation element wise to current matrix. Two matrices must be of same size.
 * \param rhs The matrix which is used in addition to current matrix.
 */
template <typename T>
const Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& rhs)
{
   if(m_rows != rhs.m_rows || m_cols != rhs.m_cols)
      SKEPU_ERROR("ERROR: Matrix should be of same size for this operation!");

   rhs.updateHost();
   updateHostAndInvalidateDevice();

   for(size_type r=0; r<m_rows; r++)
      for(size_type c=0; c<m_cols; c++)
      {
         m_data[r*m_cols+c] += rhs.m_data[r*m_cols+c];
      }
   return *this;
}


/*!
 *  Adds a scalar value to all elements in the current matrix.
 * \param rhs The value which is used in addition to current matrix.
 */
template <typename T>
const Matrix<T>& Matrix<T>::operator+=(const T& rhs)
{
   updateHostAndInvalidateDevice();

   for(size_type r=0; r<m_rows; r++)
      for(size_type c=0; c<m_cols; c++)
      {
         m_data[r*m_cols+c] += rhs;
      }
   return *this;
}

/*!
 *  Subtract \p rhs matrix operation element wise to current matrix. Two matrices must be of same size.
 * \param rhs The matrix which is used in subtraction to current matrix.
 */
template <typename T>
const Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& rhs)
{
   rhs.updateHost();
   updateHostAndInvalidateDevice();

   if(m_rows != rhs.m_rows || m_cols != rhs.m_cols)
      SKEPU_ERROR("ERROR: Matrix should be of same size for this operation!");

   for(size_type r=0; r<m_rows; r++)
      for(size_type c=0; c<m_cols; c++)
      {
         m_data[r*m_cols+c] -= rhs.m_data[r*m_cols+c];
      }
   return *this;
}

/*!
 *  Subtracts a scalar value to all elements in the current matrix.
 * \param rhs The value which is used in subtraction to current matrix.
 */
template <typename T>
const Matrix<T>& Matrix<T>::operator-=(const T& rhs)
{
   updateHostAndInvalidateDevice();

   for(size_type r=0; r<m_rows; r++)
      for(size_type c=0; c<m_cols; c++)
      {
         m_data[r*m_cols+c] -= rhs;
      }
   return *this;
}


/*!
 *  Multiplies \p rhs matrix operation element wise to current matrix. Two matrices must be of same size. NB it is not matrix multiplication
 * \param rhs The matrix which is used in multiplication to current matrix.
 */
template <typename T>
const Matrix<T>& Matrix<T>::operator*=(const Matrix<T>& rhs)
{
   rhs.updateHost();
   updateHostAndInvalidateDevice();

   if(m_rows != rhs.m_rows || m_cols != rhs.m_cols)
      SKEPU_ERROR("ERROR: Matrix should be of same size for this operation!");

   for(size_type r=0; r<m_rows; r++)
      for(size_type c=0; c<m_cols; c++)
      {
         m_data[r*m_cols+c] *= rhs.m_data[r*m_cols+c];
      }
   return *this;
}

/*!
 *  Multiplies a scalar value to all elements in the current matrix.
 * \param rhs The value which is used in multiplication to current matrix.
 */
template <typename T>
const Matrix<T>& Matrix<T>::operator*=(const T& rhs)
{
   updateHostAndInvalidateDevice();

   for(size_type r=0; r<m_rows; r++)
      for(size_type c=0; c<m_cols; c++)
      {
         m_data[r*m_cols+c] *= rhs;
      }
   return *this;
}

/*!
 *  Divides \p rhs matrix operation element wise to current matrix. Two matrices must be of same size. NB it is not matrix multiplication
 * \param rhs The matrix which is used in division to current matrix.
 */
template <typename T>
const Matrix<T>& Matrix<T>::operator/=(const Matrix<T>& rhs)
{
   rhs.updateHost();
   updateHostAndInvalidateDevice();

   if(m_rows != rhs.m_rows || m_cols != rhs.m_cols)
      SKEPU_ERROR("ERROR: Matrix should be of same size for this operation!");

   for(size_type r=0; r<m_rows; r++)
      for(size_type c=0; c<m_cols; c++)
      {
         m_data[r*m_cols+c] /= rhs.m_data[r*m_cols+c];
      }
   return *this;
}

/*!
 *  Divides a scalar value to all elements in the current matrix.
 * \param rhs The value which is used in division to current matrix.
 */
template <typename T>
const Matrix<T>& Matrix<T>::operator/=(const T& rhs)
{
   updateHostAndInvalidateDevice();

   for(size_type r=0; r<m_rows; r++)
      for(size_type c=0; c<m_cols; c++)
      {
         m_data[r*m_cols+c] /= rhs;
      }
   return *this;
}


/*!
 *  Taking Mod with \p rhs matrix, element wise to current matrix. Two matrices must be of same size.
 * \param rhs The value which is used in taking mod to current matrix.
 */
template <typename T>
const Matrix<T>& Matrix<T>::operator%=(const Matrix<T>& rhs)
{
   rhs.updateHost();
   updateHostAndInvalidateDevice();
   if(m_rows != rhs.m_rows || m_cols != rhs.m_cols)
      SKEPU_ERROR("ERROR: Matrix should be of same size for this operation!");

   for(size_type r=0; r<m_rows; r++)
      for(size_type c=0; c<m_cols; c++)
      {
         m_data[r*m_cols+c] %= rhs.m_data[r*m_cols+c];
      }
   return *this;
}

/*!
 *  Taking Mod with a scalar value to all elements in the current matrix.
 * \param rhs The value which is used in taking mod to current matrix.
 */
template <typename T>
const Matrix<T>& Matrix<T>::operator%=(const T& rhs)
{
   updateHostAndInvalidateDevice();

   for(size_type r=0; r<m_rows; r++)
      for(size_type c=0; c<m_cols; c++)
      {
         m_data[r*m_cols+c] %= rhs;
      }
   return *this;
}


///////////////////////////////////////////////
// Operators END
///////////////////////////////////////////////

///////////////////////////////////////////////
// Public Helpers START
///////////////////////////////////////////////

/*!
    *  Updates the matrix from its device allocations.
    */
template <typename T>
inline void Matrix<T>::updateHost(bool) const {}

/*!
 *  Invalidates (mark copies data invalid) all device data that this matrix has allocated.
 */
template <typename T>
inline void Matrix<T>::invalidateDeviceData(bool) const {}

/*!
 *  First updates the matrix from its device allocations. Then invalidates (mark copies data invalid) the data allocated on devices.
 */
template <typename T>
inline void Matrix<T>::updateHostAndInvalidateDevice() {}

/*!
 *  Removes the data copies allocated on devices.
 */
template <typename T>
inline void Matrix<T>::releaseDeviceAllocations() {}

/*!
 *  First updates the matrix from its device allocations. Then removes the data copies allocated on devices.
 */
template <typename T>
inline void Matrix<T>::updateHostAndReleaseDeviceAllocations() {}




///////////////////////////////////////////////
// Regular interface functions START
///////////////////////////////////////////////

//Iterators
/*!
 *  Please refer to the documentation of \p std::vector and \p skepu::Matrix::iterator.
 */
template <typename T>
typename Matrix<T>::iterator Matrix<T>::begin()
{
   return iterator(this, m_data.begin());
}

/*!
 *  Please refer to the documentation of \p std::vector and \p skepu::Matrix::iterator. Uses \p row to get an iterator for that row.
 * \param row The index of row from where to start iterator.
 */
template <typename T>
typename Matrix<T>::iterator Matrix<T>::begin(size_t row)
{
   if(row>=total_rows())
   {
      SKEPU_ERROR("ERROR! Row index is out of bound!");
   }
   return iterator(this, m_data.begin()+(row*total_cols()));
}

/*!
 *  Please refer to the documentation of \p std::vector and \p skepu::Matrix::iterator.
 */
template <typename T>
typename Matrix<T>::const_iterator Matrix<T>::begin() const
{
   return const_iterator(this, m_data.begin());
}

/*!
 *  Please refer to the documentation of \p std::vector and \p skepu::Matrix::iterator. Uses \p row to get an iterator for that row.
 * \param row The index of row from where to start iterator.
 */
template <typename T>
typename Matrix<T>::const_iterator Matrix<T>::begin(size_t row) const
{
   if(row>=total_rows())
   {
      SKEPU_ERROR("ERROR! Row index is out of bound!");
   }
   return iterator(this, m_data.begin()+(row*total_cols()));
}


/*!
 *  Please refer to the documentation of \p std::vector and \p skepu::Matrix::iterator.
 */
template <typename T>
typename Matrix<T>::iterator Matrix<T>::end()
{
   return iterator(this, m_data.end());
}

/*!
 *  Please refer to the documentation of \p std::vector and \p skepu::Matrix::iterator. Get iterator to last element of \p row.
 * \param row Index of row the iterator will point to the last element.
 */
template <typename T>
typename Matrix<T>::iterator Matrix<T>::end(size_t row)
{
   if(row>=total_rows())
   {
      SKEPU_ERROR("ERROR! Row index is out of bound!");
   }
   return iterator(this, m_data.end()-((total_rows()-(row+1))*total_cols()));
}

/*!
 *  Please refer to the documentation of \p std::vector and \p skepu::Matrix::iterator.
 */
template <typename T>
typename Matrix<T>::const_iterator Matrix<T>::end() const
{
   return const_iterator(this, m_data.end());
}

/*!
 *  Please refer to the documentation of \p std::vector and \p skepu::Matrix::iterator. Get iterator to last element of \p row.
 * \param row Index of row the iterator will point to the last element.
 */
template <typename T>
typename Matrix<T>::const_iterator Matrix<T>::end(size_t row) const
{
   if(row>=total_rows())
   {
      SKEPU_ERROR("ERROR! Row index is out of bound!");
   }
   return iterator(this, m_data.end()-((total_rows()-(row+1))*total_cols()));
}


/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
typename Matrix<T>::size_type Matrix<T>::capacity() const
{
   return m_data.capacity();
}



/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
bool Matrix<T>::empty() const
{
   return m_data.empty();
}





/*!
 *  To initialize a matrix with soem scalar value.
 *
 *  \param elem The element you want to assign to all matrix.
 */
template <typename T>
Matrix<T>& Matrix<T>::operator=(const T& elem)
{
   for(size_type i=0; i<size(); i++)
   {
      m_data[i]= elem;
   }
   return *this;
}


/*!
 *  Please refer to the documentation of \p std::vector. Uses \p row and \p col instead of single index.
 *  \param row Index of row to get.
 *  \param col Index of column to get.
 *  \return a const reference to T element at position identified by row,column index.
 */
template <typename T>
T& Matrix<T>::at(size_type row, size_type col )
{
   updateHost();
   if(row >= this->total_rows() || col >= this->total_cols())
      SKEPU_ERROR("ERROR! Row or Column index is out of bound!");

   return m_data.at(row*m_cols+col);
}

/*!
 *  To get a subsection of matrix. This will creat a separate copy.
 *  \param row Index of row to get.
 * \param rowWidth Width of the row of new Matrix.
 *  \param col Index of column to get.
 * \param colWidth Width of column of new Matrix.
 */
template <typename T>
Matrix<T>& Matrix<T>::subsection(size_type row, size_type col, size_type rowWidth, size_type colWidth)
{
   updateHost();

   if(row+rowWidth>= total_rows())
      SKEPU_ERROR("ERROR! row index and width is larger than total rows!");

   if(col+colWidth>= total_cols())
      SKEPU_ERROR("ERROR! column index and column width is larger than total columns!");


   Matrix<T> *submat=new Matrix<T>(rowWidth, colWidth);
   std::cout<<submat->total_rows()<<"   "<<submat->total_cols();
   for(typename Matrix<T>::size_type r=row, rsub=0; rsub<rowWidth; r++, rsub++)
   {
      for(typename Matrix<T>::size_type c=col, csub=0; csub<colWidth; c++, csub++)
      {
         submat->at(rsub,csub)= this->at(r,c);
      }
   }
   return *submat;
}

/*!
 *  Return index of last element of \p row.
 *
 * \param row Index of the row.
 */
template <typename T>
typename Matrix<T>::size_type Matrix<T>::row_back(size_type row)
{
   if(row>=m_rows)
      SKEPU_ERROR("Row index out of bound exception");

   updateHost();
   typename Matrix<T>::size_type index= ((row+1) * m_cols)-1;
   return index;
}



/*!
 *  Return last element of \p row.
 *
 * \param row Index of the row.
 */
template <typename T>
const T& Matrix<T>::row_back(size_type row) const
{
   if(row>=m_rows)
      SKEPU_ERROR("Row index out of bound exception");
   updateHost();
   typename Matrix<T>::size_type index= ((row+1) * m_cols)-1;
   return m_data[index];
}

/*!
 *  Return index of first element of \p row in 1D container.
 *
 * \param row Index of the row.
 */
template <typename T>
typename Matrix<T>::size_type Matrix<T>::row_front(size_type row)
{
   if(row>=m_rows)
      SKEPU_ERROR("Row index out of bound exception");
   updateHost();
   return (row * m_cols);
}

/*!
 *  Return first element of \p row.
 *
 * \param row Index of the row.
 */
template <typename T>
const T& Matrix<T>::row_front(size_type row) const
{
   if(row>=m_rows)
      SKEPU_ERROR("Row index out of bound exception");

   updateHost();
   return m_data[(row * m_cols)];
}



/*!
 *  Returns last element in \p column.
 *
 * \p col Index of the column.
 */
template <typename T>
T& Matrix<T>::col_back(size_type col)
{

   if(col>=m_cols)
      SKEPU_ERROR("Column index out of bound exception");

   updateHost();
   typename Matrix<T>::size_type index= ((m_rows-1)*m_cols)+col;
   return m_data[index];
}



/*!
 *  Returns last element in \p column.
 *
 * \p col Index of the column.
 */
template <typename T>
T& Matrix<T>::col_front(size_type col)
{

   if(col>=m_cols)
      SKEPU_ERROR("Column index out of bound exception");
   updateHost();
   return m_data[col];
}


/*!
 *  Please refer to the documentation of \p std::vector.
 * Invalidates all copies before clear.
 */
template <typename T>
void Matrix<T>::clear()
{
   invalidateDeviceData();

   m_data.clear();
}

/*!
 *  Please refer to the documentation of \p std::vector.
 * Updates and invalidate both Matrices before swapping.
 */
template <typename T>
void Matrix<T>::swap(Matrix<T>& from)
{
   updateHostAndInvalidateDevice();
   from.updateHostAndInvalidateDevice();

   item_swap<typename Matrix<T>::size_type>(m_rows, from.m_rows);
   item_swap<typename Matrix<T>::size_type>(m_cols, from.m_cols);
   item_swap<typename Matrix::container_type>(m_data, from.m_data);
}


/*!
 *  Please refer to the documentation of \p std::vector.
 * Updates and invalidate the Matrix.
 */
template <typename T>
typename Matrix<T>::iterator Matrix<T>::erase( typename Matrix<T>::iterator loc )
{
   updateHostAndInvalidateDevice();

   return iterator(m_data.erase(loc), *this);
}

/*!
 *  Please refer to the documentation of \p std::vector.
 * Erases a certain number of elements pointed by \p start and \p end. Updates and Invalidates all copies before.
 */
template <typename T>
typename Matrix<T>::iterator Matrix<T>::erase( typename Matrix<T>::iterator start, typename Matrix<T>::iterator end )
{
   updateHostAndInvalidateDevice();

   return iterator(m_data.erase(start, end), *this);
}



///////////////////////////////////////////////
// Regular interface functions END
///////////////////////////////////////////////


///////////////////////////////////////////////
// Additions to interface START
///////////////////////////////////////////////


/*!
 *  Flushes the matrix, synchronizing it with the device then release all device allocations.
 */
template <typename T>
void Matrix<T>::flush()
{
#ifdef SKEPU_OPENCL
   flush_CL();
#endif

#ifdef SKEPU_CUDA
   flush_CU();
#endif
}


/*!
 *  Behaves like \p operator[] and unlike \p skepu::Vector, it cares about synchronizing with device.
 *  Can be used when accessing to access elements row and column wise.
 *
 *  \param row Index to a specific row of the Matrix.
 *  \param col Index to a specific column of the Matrix.
 */
template <typename T>
const T& Matrix<T>::operator()(const size_type row, const size_type col) const
{
   updateHost();
   if(row >= this->total_rows() || col >= this->total_cols())
      SKEPU_ERROR("ERROR! Row or Column index is out of bound!");
   return m_data[row * m_cols + col];
}


/*!
 *  Behaves like \p operator[] and unlike \p skepu::Vector, it cares about synchronizing with device.
 *  Can be used when accessing to access elements row and column wise.
 *
 *  \param row Index to a specific row of the Matrix.
 *  \param col Index to a specific column of the Matrix.
 */
template <typename T>
T& Matrix<T>::operator()(const size_type row, const size_type col)
{
   updateHostAndInvalidateDevice();
   if(row >= this->total_rows() || col >= this->total_cols())
      SKEPU_ERROR("ERROR! Row or Column index is out of bound!");
   return m_data[row * m_cols + col];
}

/*!
 *  Behaves like \p operator[] but does not care about synchronizing with device.
 *  Can be used when accessing many elements quickly so that no synchronization
 *  overhead effects performance. Make sure to properly synch with device by calling
 *  updateHost etc before use.
 *
 *  \param index Index of element assuming continuous Matrix row-wise storage. To facilitate access using single indexing
 */
template <typename T>
T& Matrix<T>::operator()(const size_type index)
{
   return m_data[index];
}

/*!
 *  Behaves like \p operator[] but does not care about synchronizing with device.
 *  Can be used when accessing many elements quickly so that no synchronization
 *  overhead effects performance. Make sure to properly synch with device by calling
 *  updateHost etc before use.
 *
 *  \param index Index of element assuming continuous Matrix row-wise storage. To facilitate access using single indexing
 */
template <typename T>
T& Matrix<T>::operator()(const Index2D index)
{
   return m_data[index.row * m_cols + index.col];
}

/*!
 *  A \p operator[] that care about synchronizing with device.
 *  Can be used when accessing elements considering consecutive storage
 *
 *  \param index Index of element assuming continuous Matrix row-wise storage. To facilitate access using single indexing
 */
template <typename T>
const T& Matrix<T>::operator[](const size_type index) const
{
   updateHost();
   if(index >= (this->total_rows() * this->total_cols()))
      SKEPU_ERROR("ERROR! Index is out of bound!");
   return m_data[index];
}

/*!
 *  A \p operator[] that care about synchronizing with device.
 *  Can be used when accessing elements considering consecutive storage
 *
 *  \param index Index of element assuming continuous Matrix row-wise storage. To facilitate access using single indexing
 */
template <typename T>
T& Matrix<T>::operator[](const size_type index)
{
   updateHostAndInvalidateDevice();
   if(index >= (this->total_rows() * this->total_cols()))
      SKEPU_ERROR("ERROR! Index is out of bound!");
   return m_data[index];
}

///////////////////////////////////////////////
// Additions to interface END
///////////////////////////////////////////////


///////////////////////////////////////////////
// Comparison operators START
///////////////////////////////////////////////


/*!
 *  Please refer to the documentation of \p std::vector.
 *
 */
template <typename T>
bool Matrix<T>::operator==(const Matrix<T>& c1)
{
   c1.updateHost();
   updateHost();

   return (c1.m_data == m_data);
}

/*!
 *  Please refer to the documentation of \p std::vector.
 *
 */
template <typename T>
bool Matrix<T>::operator!=(const Matrix<T>& c1)
{
   c1.updateHost();
   updateHost();

   return (c1.m_data != m_data);
}


/*!
 *  Please refer to the documentation of \p std::vector.
 *
 */
template <typename T>
bool Matrix<T>::operator<(const Matrix<T>& c1)
{
   c1.updateHost();
   updateHost();

   return (c1.m_data < m_data);
}

/*!
 *  Please refer to the documentation of \p std::vector.
 *
 */
template <typename T>
bool Matrix<T>::operator>(const Matrix<T>& c1)
{
   c1.updateHost();
   updateHost();

   return (c1.m_data > m_data);
}

/*!
 *  Please refer to the documentation of \p std::vector.
 *
 */
template <typename T>
bool Matrix<T>::operator<=(const Matrix<T>& c1)
{
   c1.updateHost();
   updateHost();

   return (c1.m_data <= m_data);
}


/*!
 *  Please refer to the documentation of \p std::vector.
 *
 */
template <typename T>
bool Matrix<T>::operator>=(const Matrix<T>& c1)
{
   c1.updateHost();
   updateHost();

   return (c1.m_data >= m_data);
}


} // end namespace skepu


#include "matrix_iterator.inl"