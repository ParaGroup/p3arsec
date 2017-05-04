/*! \file sparse_matrix.inl
 *  \brief Contains the definitions of member functions of the SparseMatrix class that are not related to any backend.
 */

namespace skepu2
{


// **********************************************************************************//
// **********************************************************************************//
// ------------------------    CONSTRUCTORS/DESTRUCTORS   ---------------------------//
// **********************************************************************************//
// **********************************************************************************//


/*!
 *  SparseMatrix Constructor, used to create a sparse matrix with given data (rows-offsets,cols-indices and non-zero elements).
 * \param rows Number of rows in the sparse matrix.
 * \param cols Number of columns in the sparse matrix.
 * \param nnz Number of non-zero elements in the sparse matrix.
 * \param values An array containing non-zero elements stored row-wise in C order.
 * \param rowPtr An array containing indices pointing to starting indices of each row in the \em values array.
 * \param colInd An array containing indices pointing to column indices for each element in the \em values array.
 * \param dealloc A boolean defining whether the arrays are going to be de-allocated when the destructor is called.
 * \param zeroValue value that represent zero value for the given elements type, default will be initial value of that data type.
 * \param transMatrix A boolean that specifies whether the matrix is a transpose matrix or a normal one.
 */
template<typename T>
SparseMatrix<T>::SparseMatrix(size_t rows, size_t cols, size_t nnz, T *values, size_t *rowPtr, size_t *colInd, bool dealloc, T zeroValue, bool transMatrix): m_rows(rows), m_cols(cols), m_nnz(nnz), m_values(NULL), m_rowPtr(NULL), m_colInd(NULL), m_dealloc(dealloc), m_zeroValue(zeroValue), m_transposeValid(false), m_colPtrCSC(NULL), m_rowIndCSC(NULL), m_valuesCSC(NULL), m_cscMatrix(NULL), m_transMatrix(transMatrix)
{

#if defined(SKEPU_CUDA) && defined(USE_PINNED_MEMORY)
   if(!m_transMatrix) // if not a transposed matrix then perform this check.
   {
      SKEPU_WARNING("Pinned memory allocation enabled. Will create a copy of values array to use internally.\n");
      backend::allocateHostMemory<T>(m_values, m_nnz); // can be pinned if enabled
      std::copy(&values[0], &values[m_nnz], m_values);

      backend::allocateHostMemory<size_t>(m_rowPtr, m_rows+1);
      std::copy(&(rowPtr[0]), &(rowPtr[m_rows]), m_rowPtr);

      backend::allocateHostMemory<size_t>(m_colInd, m_nnz);
      std::copy(&(colInd[0]), &(colInd[m_nnz]), m_colInd);
   }
   else  // otherwise use the same memory.
   {
      m_values = values;
      m_rowPtr = rowPtr;
      m_colInd = colInd;
   }
#else   // otherwise use the same memory.
   m_values = values;
   m_rowPtr = rowPtr;
   m_colInd = colInd;
#endif

   if(m_rows<1 || m_nnz<m_rows)
   {
      SKEPU_ERROR("The operands to sparse matrix constructor are invalid.\n");
   }
}


/*!
 *  SparseMatrix Constructor, used to generate a random sparse matrix (in privided min max range) with given rows,cols and non-zero elements.
 * \param rows Number of rows in the sparse matrix.
 * \param cols Number of columns in the sparse matrix.
 * \param nnz Number of non-zero elements in the sparse matrix.
 * \param min Minimum value for the random values generated
 * \param max Maximum value for the random values generated
 * \param zeroValue value that represent zero value for the given elements type, default will be initial value of that data type.
 */
template<typename T>
SparseMatrix<T>::SparseMatrix(size_t rows, size_t cols, size_t nnz, T min, T max, T zeroValue): m_rows(rows), m_cols(cols), m_nnz(nnz), m_values(NULL), m_rowPtr(NULL), m_colInd(NULL), m_dealloc(true), m_zeroValue(zeroValue), m_transposeValid(false), m_colPtrCSC(NULL), m_rowIndCSC(NULL), m_valuesCSC(NULL), m_cscMatrix(NULL), m_transMatrix(false)
{
   if(m_rows<2 || m_cols<2)
   {
      SKEPU_ERROR("Error: Minimux sparse matrix size can be 2 X 2 with atleast 1 non-empth element per row\nm_rows: "<<m_rows<<", m_cols: "<<m_cols<<", m_nnz: "<<m_nnz<<"\n");
   }
   if(m_nnz<m_rows)
   {
      SKEPU_ERROR("Error: Atleast 1 non-empty element must be specified per row\nm_rows: "<<m_rows<<", m_cols: "<<m_cols<<", m_nnz: "<<m_nnz<<"\n");
   }

   if(m_nnz>(m_rows*m_cols*0.5))
   {
      SKEPU_ERROR("Error: The sparse matrix should contain at most 50% non-zero elements.\nm_rows: "<<m_rows<<", m_cols: "<<m_cols<<", m_nnz: "<<m_nnz<<"\n");
   }

   backend::allocateHostMemory<T>(m_values, m_nnz); // can be pinned if enabled

   backend::allocateHostMemory<size_t>(m_colInd, m_nnz);
   backend::allocateHostMemory<size_t>(m_rowPtr, m_rows+1);

   if(!m_values || !m_colInd || !m_rowPtr)
   {
      SKEPU_ERROR("Error: Memory allocation failed for sparse matrix\nm_rows: "<<m_rows<<", m_cols: "<<m_cols<<", m_nnz: "<<m_nnz<<"\n");
   }

   srand(10);

   size_t elemPerRow= m_nnz/m_rows;
   size_t remPerRow= m_nnz%m_rows;

   size_t k=0;

   std::set<size_t> colSet;
   std::set<size_t>::iterator colIter;

   size_t i=0;
   for(; i<m_rows && k<nnz; i++)
   {
      m_rowPtr[i]=k;

      size_t totalElems = (i<remPerRow)? (elemPerRow+1): elemPerRow;

      size_t j=0;
      while(j<totalElems)
      {
         size_t colIdx= backend::get_random_number((size_t)0, m_cols-1);
         colIter = colSet.find(colIdx);

         if(colIter==colSet.end()) // this index is not generated before
         {
            colSet.insert(colIdx); // set remains sorted
            j++;
         }
      }

      // not the column indices are generated
      for(colIter=colSet.begin(); colIter!=colSet.end() && k<nnz; colIter++)
      {
         m_colInd[k] = *colIter;
         m_values[k] = backend::get_random_number(min, max);
         k++;
      }

      colSet.clear();
   }

   m_rowPtr[i]=k; // insert one off by last row-index
}



/*!
 *  SparseMatrix Constructor, used to read a sparse matrix from a text file (preferably in MTX format) with given rows,cols and non-zero elements.
 * \param inputfile Name of the input file.
 * \param format Format of the matrix storage, currently only MATRIX_MARKET_FORMAT is supported.
 * \param zeroValue value that represent zero value for the given elements type, default will be initial value of that data type.
 */
template<typename T>
SparseMatrix<T>::SparseMatrix(const std::string &inputfile, enum SparseFileFormat format, T zeroValue): m_rows(0), m_cols(0), m_nnz(0), m_values(NULL), m_rowPtr(NULL), m_colInd(NULL), m_dealloc(true), m_zeroValue(zeroValue), m_transposeValid(false), m_colPtrCSC(NULL), m_rowIndCSC(NULL), m_valuesCSC(NULL), m_cscMatrix(NULL), m_transMatrix(false)
{
   if(format==MATRIX_MARKET_FORMAT)
      readMTXFile(inputfile);
}





/*!
 *  SparseMatrix Copy Constructor, used to create a copy of another sparse matrix. Does not copy the transpose matrix part.
 * \param copy sparse matrix which we are aopying from.
 */
template<typename T>
SparseMatrix<T>::SparseMatrix(const SparseMatrix<T> &copy): m_rows(copy.m_rows), m_cols(copy.m_cols), m_nnz(copy.m_nnz), m_dealloc(true), m_zeroValue(copy.m_zeroValue), m_transposeValid(false), m_colPtrCSC(NULL), m_rowIndCSC(NULL), m_valuesCSC(NULL), m_cscMatrix(NULL), m_transMatrix(false)
{
   backend::allocateHostMemory<T>(m_values, m_nnz); // can be pinned if enabled

   backend::allocateHostMemory<size_t>(m_colInd, m_nnz); // can be pinned if enabled
   backend::allocateHostMemory<size_t>(m_rowPtr, m_rows+1); // can be pinned if enabled

   if(copy.m_values)
      std::copy(copy.m_values, copy.m_values + m_nnz, m_values);

   if(copy.m_rowPtr)
      std::copy(copy.m_rowPtr, copy.m_rowPtr + m_rows + 1, m_rowPtr);

   if(copy.m_colInd)
      std::copy(copy.m_colInd, copy.m_colInd + m_nnz, m_colInd);
     
     std::cout << "COPIED\n";
}


/*!
 *  SparseMatrix Copy Constructor, used to create a copy of another sparse matrix. Does not copy the transpose matrix part.
 * \param copy sparse matrix which we are aopying from.
 */
template<typename T>
void SparseMatrix<T>::operator=(const SparseMatrix<T> &other)
{
   if(other == *this)
      return;
   
   if(m_nnz!=other.m_nnz)
   {
      if(m_values)
         backend::deallocateHostMemory<T>(m_values);
      
      if(m_colInd)
         backend::deallocateHostMemory<T>(m_colInd);

      m_nnz = other.m_nnz;

      backend::allocateHostMemory<T>(m_values, m_nnz);
      backend::allocateHostMemory<size_t>(m_colInd, m_nnz);
   }
   if(m_rows!=other.m_rows)
   {
      if(m_rowPtr)
         backend::deallocateHostMemory<T>(m_rowPtr);
      
      m_rows = other.m_rows;

      backend::allocateHostMemory<T>(m_rowPtr, m_rows+1);
   }
   m_cols = other.m_cols;
   m_dealloc = true;
   m_zeroValue = other.m_zeroValue;
   m_transposeValid = false;
   m_colPtrCSC = NULL;
   m_rowIndCSC = NULL;
   m_valuesCSC = NULL;
   m_cscMatrix = NULL;
   m_transMatrix = false;
   
   if(other.m_values)
      std::copy(&(other.m_values[0]), &(other.m_values[m_nnz]), m_values);

   if(other.m_rowPtr)
      std::copy(&(other.m_rowPtr[0]), &(other.m_rowPtr[m_rows]), m_rowPtr);

   if(other.m_colInd)
      std::copy(&(other.m_colInd[0]), &(other.m_colInd[m_nnz]), m_colInd);
}



/*!
 *  SparseMatrix Destructor. Internally Deallocates memory if proper flags are set.
 */
template<typename T>
SparseMatrix<T>::~SparseMatrix()
{
   DEBUG_TEXT_LEVEL2("SPARSE MATRIX DESTRUCTOR CALLED")

   if(!m_transMatrix)
   {
      if(m_dealloc)
      {
         if(m_values!=NULL)
         {
            backend::deallocateHostMemory<T>(m_values);
         }
         if(m_rowPtr!=NULL)
            backend::deallocateHostMemory<size_t>(m_rowPtr);
         if(m_colInd!=NULL)
            backend::deallocateHostMemory<size_t>(m_colInd);
      }

      if(m_colPtrCSC!=NULL)
         delete[] m_colPtrCSC;
      if(m_rowIndCSC!=NULL)
         delete[] m_rowIndCSC;
      if(m_valuesCSC!=NULL)
         delete[] m_valuesCSC;
   }

#ifdef SKEPU_OPENCL
   releaseDeviceAllocations_CL();
#endif

#ifdef SKEPU_CUDA
   releaseDeviceAllocations_CU();
#endif

   if(m_cscMatrix!=NULL)
      delete m_cscMatrix;
}



// **********************************************************************************//
// **********************************************************************************//
// ----------------------------    REGULAR FUNCTIONS   ------------------------------//
// **********************************************************************************//
// **********************************************************************************//

/*!
 *  Uses \p row and \p col instead to find element. If found, return actual otherwise returns 0
 *  \param row Index of row to get.
 *  \param col Index of column to get.
 */
template <typename T>
T SparseMatrix<T>::at(size_t row, size_t col ) const
{
   updateHost();

   if(SKEPU_UNLIKELY(row>=total_rows()))
   {
      SKEPU_ERROR("Out of bound access in sparse matrix, total rows: "<<total_rows()<<", accessed row: "<<row<<"\n");
   }

   if(SKEPU_UNLIKELY(col>=total_cols()))
   {
      SKEPU_ERROR("Out of bound access in sparse matrix, total cols: "<<total_cols()<<", accessed col: "<<col<<"\n");
   }

   size_t rowIdx = m_rowPtr[row];
   size_t nxtRowIdx = m_rowPtr[row+1];

   for(size_t jj=rowIdx; jj<nxtRowIdx; jj++)
   {
      if(m_colInd[jj]==col)
         return m_values[jj];
   }
   return m_zeroValue;
}


/*!
 *  Directly access an element by specifying its index. If found, return actual otherwise returns 0
 *  \param index Index of row to get.
 */
template <typename T>
const T& SparseMatrix<T>::at(size_t index) const
{
   updateHost();

   if(SKEPU_UNLIKELY(index>=total_nnz()))
   {
      SKEPU_ERROR("Out of bound access in sparse matrix, total values: "<<total_nnz()<<", accessed value: "<<index<<"\n");
   }

   return m_values[index];
}



/*!
 *  Please refer to the documentation of \p std::vector and \p skepu::Matrix::iterator. Uses \p row to get an iterator for that row.
 * \param row The index of row from where to start iterator.
 */
template <typename T>
inline typename SparseMatrix<T>::iterator SparseMatrix<T>::begin(unsigned row)
{
   if(row>=total_rows())
   {
      SKEPU_ERROR("Row index is out of bound!\n");
   }

   return iterator(this, &m_values[m_rowPtr[row]], &m_values[m_rowPtr[row+1]]);
}




/*!
 *  Can be used to access elements by specifying row and column index. it cares about synchronizing with device.
 *  Can be used when accessing to access elements row and column wise.
 *
 *  \param row Index to a specific row of the sparse matrix.
 *  \param col Index to a specific column of the sparse matrix.
 */
template <typename T>
const T& SparseMatrix<T>::operator()(const size_t row, const size_t col) const
{
   updateHost();

   if(SKEPU_UNLIKELY(row>=total_rows()))
   {
      SKEPU_ERROR("Out of bound access in sparse matrix, total rows: "<<total_rows()<<", accessed row: "<<row<<"\n");
   }

   if(SKEPU_UNLIKELY(col>=total_cols()))
   {
      SKEPU_ERROR("Out of bound access in sparse matrix, total cols: "<<total_cols()<<", accessed col: "<<col<<"\n");
   }

   size_t rowIdx = m_rowPtr[row];
   size_t nxtRowIdx = m_rowPtr[row+1];

   for(size_t jj=rowIdx; jj<nxtRowIdx; jj++)
   {
      if(m_colInd[jj]==col)
         return m_values[jj];
   }
   return m_zeroValue;
}


/*!
 *  Behaves like \p operator[] and unlike \p skepu::Vector, it cares about synchronizing with device.
 *  Can be used when accessing to access elements row and column wise.
 *
 *  \param row Index to a specific row of the sparse matrix.
 *  \param col Index to a specific column of the sparse matrix.
 */
template <typename T>
T SparseMatrix<T>::operator()(const size_t row, const size_t col)
{
   updateHost();

   if(SKEPU_UNLIKELY(row>=total_rows()))
   {
      SKEPU_ERROR("Out of bound access in sparse matrix, total rows: "<<total_rows()<<", accessed row: "<<row<<"\n");
   }

   if(SKEPU_UNLIKELY(col>=total_cols()))
   {
      SKEPU_ERROR("Out of bound access in sparse matrix, total cols: "<<total_cols()<<", accessed col: "<<col<<"\n");
   }

   size_t rowIdx = m_rowPtr[row];
   size_t nxtRowIdx = m_rowPtr[row+1];

   for(size_t jj=rowIdx; jj<nxtRowIdx; jj++)
   {
      if(m_colInd[jj]==col)
         return m_values[jj];
   }
   return m_zeroValue;
}


/*!
 * will resize the matrix, can be dangerous, used with care
 */
template <typename T>
void SparseMatrix<T>::resize(SparseMatrix<T> &copy, bool retainData)
{
   bool copyData = (retainData && (m_values!=NULL));
   if(m_nnz != copy.m_nnz)
   {
      deleteCSCFormat(); // First delete CSC format if generated already

      T *tmp;
      if(copyData)
         tmp = m_values;
      else if (m_values!=NULL)
      {
         backend::deallocateHostMemory<T>(m_values);
      }

      m_values = backend::allocateHostMemory<T>(copy.m_nnz);

      if(copyData)
      {
         size_t limit = (m_nnz<copy.m_nnz) ? m_nnz:copy.m_nnz;

         std::copy(&tmp[0], &tmp[limit], m_values);

         delete tmp;
      }

      if(m_rows!=copy.m_rows)
      {
         delete m_rowPtr;
         m_rowPtr = new size_t[copy.m_rows+1];
      }

      std::copy(&(copy.m_rowPtr[0]), &(copy.m_rowPtr[copy.m_rows+1]), m_rowPtr);

      // Already checked in parent condition, so it means that colInd with be unequal as both m_colInd and m_value are of same size.....
      delete m_colInd;
      m_colInd = new size_t[copy.m_nnz];

      std::copy(&(copy.m_colInd[0]), &(copy.m_colInd[copy.m_nnz]), m_colInd);

      m_nnz=copy.m_nnz;
      m_rows = copy.m_rows;
      m_cols=copy.m_cols;
   }
}


// **********************************************************************************//
// **********************************************************************************//
// ----------------------------    HELPER FUNCTIONS   -------------------------------//
// **********************************************************************************//
// **********************************************************************************//

template <typename T>
void SparseMatrix<T>::readMTXFile(const std::string &inputfile)
{
   std::ifstream fin(inputfile.c_str());
   if(!fin.is_open())
   {
      SKEPU_ERROR("Cannot open the following input file: "<<inputfile<<"\n");
   }

   size_t rows,cols,nnz,i,j;
   std::string strLine;

   getline(fin, strLine);

   strLine = backend::trimSpaces(strLine);

   while(strLine[0]=='%')
   {
      getline(fin, strLine);
      strLine = backend::trimSpaces(strLine);
   }

   std::istringstream str(strLine);
   str >> rows >> cols;
   if(rows<1)
   {
      SKEPU_ERROR("Error while reading the rows and columns size from file: "<<inputfile<<"\n");
   }
   str >> nnz;
   if(nnz<1 && cols>1) // check when it is square matrix and size is specified for 1 dimension rather than rows and columns
   {
      nnz=cols;
      cols = rows;
   }

   
// ******* Actual read operations starts now

   std::map<size_t, std::map<size_t , T> > mat;

   size_t k=0;
   while(k<nnz)
   {
      fin >> i >> j;
      fin >> mat[i][j];
      k++;
   }

   fin.close();

// ***** allocate storage
   if(m_values != NULL && (m_nnz!=nnz || m_rows!=rows || m_cols!=cols))
   {
      backend::deallocateHostMemory<T>(m_values);
      m_values = NULL;
   }
   if(m_values==NULL)
   {
      m_nnz = nnz;
      m_rows = rows;
      m_cols = cols;

      m_values = backend::allocateHostMemory<T>(m_nnz); // can be pinned if enabled
   }

   if(m_colInd != NULL && (m_nnz!=nnz || m_rows!=rows || m_cols!=cols))
   {
      delete m_colInd;
      m_colInd = NULL;
   }
   if(m_colInd==NULL)
   {
      m_colInd = new size_t[m_nnz];
   }

   if(m_rowPtr != NULL && (m_nnz!=nnz || m_rows!=rows || m_cols!=cols))
   {
      delete m_rowPtr;
      m_rowPtr = NULL;
   }
   if(m_rowPtr==NULL)
   {
      m_rowPtr = new size_t[m_rows+1];
   }

   typename std::map<size_t, std::map<size_t , T> >::iterator ii;
   typename std::map<size_t, T>::iterator jj;

   k=0;
   size_t rowIdx = 0;
   for(ii=mat.begin(); ii!=mat.end(); ii++)
   {
      m_rowPtr[rowIdx] = k;
      rowIdx++;

      if(ii!=mat.end())
      {
         for(jj=(*ii).second.begin(); jj!=(*ii).second.end(); jj++)
         {
            if(k<m_nnz)
            {
               m_values[k] = (*jj).second;
               m_colInd[k] = (*jj).first;
            }
            k++;
         }
      }
   }
   m_rowPtr[rowIdx] = nnz;
}




// **********************************************************************************//
// **********************************************************************************//
// ------------------------    MEMORY MANAGEMENT FUNCTIONS   ------------------------//
// **********************************************************************************//
// **********************************************************************************//


/*!
 *  Updates the sparse matrix from its device allocations.
 */
template <typename T>
inline void SparseMatrix<T>::updateHost(bool enabled) const
{
   if (!enabled)
      return;
      
#ifdef SKEPU_OPENCL
   updateHost_CL();
#endif

#ifdef SKEPU_CUDA
   updateHost_CU();
#endif
}

/*!
 *  Invalidates all device data that this sparse matrix has allocated.
 */
template <typename T>
inline void SparseMatrix<T>::invalidateDeviceData(bool enabled)
{
   if (!enabled)
      return;
      
#ifdef SKEPU_OPENCL
   invalidateDeviceData_CL();
#endif

#ifdef SKEPU_CUDA
   invalidateDeviceData_CU();
#endif
}

/*!
 *  First updates the sparse matrix from its device allocations. Then invalidates the data allocated on devices.
 */
template <typename T>
inline void SparseMatrix<T>::updateHostAndInvalidateDevice()
{
#ifdef SKEPU_OPENCL
   updateHost_CL();
   invalidateDeviceData_CL();
#endif

#ifdef SKEPU_CUDA
   updateHost_CU();
   invalidateDeviceData_CU();
#endif
}







}
