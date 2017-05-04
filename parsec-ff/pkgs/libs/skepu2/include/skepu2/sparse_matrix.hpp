/*! \file sparse_matrix.h
 *  \brief Contains a class declaration for the SparseMatrix container.
 */

#ifndef _SPARSE_MATRIX_H
#define _SPARSE_MATRIX_H

#include <cstdlib>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <iomanip>
#include <set>


#include "backend/malloc_allocator.h"

#ifdef SKEPU_PRECOMPILED
#include "backend/device_mem_pointer_cl.h"
#include "backend/device_mem_pointer_cu.h"
#include "backend/environment.h"
#endif // SKEPU_PRECOMPILED

namespace skepu2
{

/*!
*  \brief Can be used to specify the input format for a sparse matrix that is supplied in constructor
*
*  Used to load sparse matrix from existing files.
*/
enum SparseFileFormat
{
   MATRIX_MARKET_FORMAT,
   MATLAB_FORMAT,
   RUTHERFOR_BOEING_FORMAT
};




template<typename T>
struct SparseMat
{
	T *data;
	size_t *row_offsets;
	size_t *col_indices;
	size_t count;
	
	T operator[](size_t index) const
	{
		return this->data[index];
	}
};


/*!
*  \class SparseMatrix
*
*  \brief A sparse matrix container class that mainly stores its data in CSR format.
*
*  A \p skepu::SparseMatrix is a container for storing sparse 2D structures that are internally stores in a 1D array using CSR format.
*  It supports operation to load sparse matrices from file as well as creating random sparse matrices.
*  As CSR format, it stores 3 arrays: actual values, column indices and row offsets. It also keeps track of which parts
*  of it are currently allocated and uploaded to the GPU.
*  If a computation is done, changing the elements in the GPU memory, it is not directly transferred back to the
*  host memory. Instead, the SparseMatrix waits until an element is accessed before any copying is done.
*  The class also supports transpose operation which it achieves by converting CSR (Compressed Storage Rows) format
*  to CSC (Compressed Storage Column) format which can be useful e.g. when applying column-wise reduction operation.
*  The transpose can be obtained by simply '~' operation, e.g., transpose_m0 = ~m0;
*
*  It also implements support for allocating and de-allocating page-locked memory using cudaMallocHost and cudaFreeHost.
*  This could help is running asynchronous operations especially when using multiple CUDA devices.
*  It can be enabled by defining USE_PINNED_MEMORY flag in the skeleton program.
*
*
*/
template <class T>
class SparseMatrix
{

// typedefs
public:
#ifdef SKEPU_CUDA
   typedef backend::DeviceMemPointer_CU<T>* device_pointer_type_cu;
   typedef backend::DeviceMemPointer_CU<size_t>* device_pointer_index_type_cu;
#endif

#ifdef SKEPU_OPENCL
   typedef backend::DeviceMemPointer_CL<T>* device_pointer_type_cl;
   typedef backend::DeviceMemPointer_CL<size_t>* device_pointer_index_type_cl;
#endif

	typedef ptrdiff_t difference_type;

private:
   T *m_values;
   size_t *m_colInd;
   size_t *m_rowPtr;
   size_t m_rows;
   size_t m_cols;
   size_t m_nnz;
   T m_zeroValue;
   bool m_dealloc;

//    bool m_valid; /*! to keep track of whether the main copy is valid or not */

   // ***** RELATED to transpose of the matrix ****/
   T *m_valuesCSC;
   size_t *m_colPtrCSC;
   size_t *m_rowIndCSC;
   bool m_transposeValid;
   SparseMatrix *m_cscMatrix;

   

public:
   class iterator;
   
   // following bool to is to control memory deallocation as we assign de-allocation responsibility to main matrix and not the transpose matrix itself.
   bool m_transMatrix;

private:

   void deleteCSCFormat() // when resizing so we need to take transpose again and also would have to allocate storgae again.
   {
      if(m_transMatrix) // do nothing then
         return;

      m_transposeValid = false;

      if(m_colPtrCSC!=NULL)
      {
         delete m_colPtrCSC;
         m_colPtrCSC = NULL;
      }
      if(m_rowIndCSC!=NULL)
      {
         delete m_rowIndCSC;
         m_rowIndCSC = NULL;
      }
      if(m_valuesCSC!=NULL)
      {
         delete m_valuesCSC;
         m_valuesCSC = NULL;
      }

      if(m_cscMatrix!=NULL)
      {
         delete m_cscMatrix;
         m_cscMatrix = NULL;
      }
   }

   void convertToCSCFormat()
   {
      if(m_transMatrix) // cannot transpose an already transposed
      {
         std::cerr<<"Cannot apply transpose operation on a matrix which is already in CSC format.\n";
         SKEPU_EXIT();
      }

      if(m_transposeValid && (m_valuesCSC!=NULL) && (m_rowIndCSC!=NULL) && (m_colPtrCSC!=NULL)) // already there
         return;

      updateHost(); // update Host for new data;

      if(m_valuesCSC==NULL)
         m_valuesCSC = new T[m_nnz];
      if(m_rowIndCSC == NULL)
         m_rowIndCSC = new size_t[m_nnz];
      if(m_colPtrCSC == NULL)
         m_colPtrCSC = new size_t[m_cols+1];

      size_t rowIdx = 0;
      size_t nxtRowIdx = 0;

      std::multimap<size_t, std::pair<size_t, T>, std::less<size_t> > cscFormat;

      for(size_t ii = 0; ii < m_rows; ii++)
      {
         rowIdx = m_rowPtr[ii];
         nxtRowIdx = m_rowPtr[ii+1];

         for(size_t jj=rowIdx; jj<nxtRowIdx; jj++)
         {
            cscFormat.insert( std::make_pair(m_colInd[jj], std::make_pair(ii, m_values[jj])) );
         }
      }

      // make multimap sorted based on key which is column index.
      typedef typename std::multimap<size_t, std::pair<size_t, T>, std::less<size_t> >::iterator mapIter;
      mapIter m_it, s_it;

      size_t col = 0, ind=0;

      m_colPtrCSC[0] = 0;

      size_t expColInd = 0; // assign it to first column index

      for (m_it = cscFormat.begin();  m_it != cscFormat.end();  m_it = s_it)
      {
         size_t colInd = (*m_it).first;

         if(expColInd < colInd) // meaning some intermediate columns are totally blank, i.e., no non-zero entries, add their record
         {
            for(size_t i=expColInd; i<colInd; i++)
            {
               m_colPtrCSC[++col] = ind;
            }
         }
         expColInd = colInd+1; // now set expected to next column.

         std::pair<mapIter, mapIter> keyRange = cscFormat.equal_range(colInd);

         // Iterate over all map elements with key == theKey
         for (s_it = keyRange.first;  s_it != keyRange.second;  ++s_it)
         {
            m_rowIndCSC[ind] = (*s_it).second.first;
            m_valuesCSC[ind] = (*s_it).second.second;
            ind++;
         }
         m_colPtrCSC[++col] = ind;
      }
      for(size_t i= (col+1); i<=m_cols; i++)
      {
         m_colPtrCSC[i] = ind;
      }

      m_transposeValid = true;
   }


   void readMTXFile(const std::string &inputfile);

public:
   
// #if SKEPU_DEBUG>0      
   std::string m_nameVerbose; // for debugging useful
// #endif

   void printMatrixInDenseFormat()
   {
      std::cerr<<"SparseMatrix ("<<m_rows <<" X "<< m_cols<<") nnz: "<<m_nnz<<"\n";

      T *temp = new T[m_cols];

      size_t rowIdx, nxtRowIdx;
      for(size_t ii = 0; ii < m_rows; ii++)
      {
         for(size_t i=0; i<m_cols; i++)
            temp[i] = T();

         rowIdx = m_rowPtr[ii];
         nxtRowIdx = m_rowPtr[ii+1];

         for(size_t jj=rowIdx; jj<nxtRowIdx; jj++)
         {
            temp[m_colInd[jj]] = m_values[jj];
         }

         for(size_t i=0; i<m_cols; i++)
            std::cerr<<std::setw(5)<<temp[i];

         std::cerr<<"\n";
      }

      delete[] temp;
   }

   // unary CSC operator
   inline SparseMatrix<T>& operator~()
   {
      convertToCSCFormat();

      if(m_cscMatrix==NULL)
      {
         m_cscMatrix = new SparseMatrix(m_cols, m_rows, m_nnz, m_valuesCSC, m_colPtrCSC, m_rowIndCSC, false, T(), true);
      }

      return *m_cscMatrix;
   }

   // ----- CONSTRUCTORS & DESTRUCTORS -------//
   SparseMatrix(size_t rows, size_t cols, size_t nnz, T *values, size_t *rowPtr, size_t *colInd, bool dealloc=true, T zeroValue=T(), bool transMatrix=false);

   SparseMatrix(size_t rows, size_t cols, size_t nnz, T min, T max, T zeroValue=T());

   SparseMatrix(const SparseMatrix &copy);

   SparseMatrix(const std::string &inputfile, enum SparseFileFormat format=MATRIX_MARKET_FORMAT, T zeroValue=T());

   ~SparseMatrix();
   
   void operator=(const SparseMatrix<T> &copy);
   
   
   const SparseMatrix<T>& getParent() const
   {
      return *this;
   }
   
   SparseMatrix<T>& getParent()
   {
      return *this;
   }

   // ----- FRIEND METHODS       -------//

   /*!
    *  \brief Overloaded stream operator, for testing purposes.
    *
    *  Outputs the sparse matrix having one element on each line.
    */
   friend std::ostream& operator<<(std::ostream &os, SparseMatrix<T>& matrix)
   {
      matrix.updateHost();

      os<<"Matrix rows="<< matrix.total_rows() <<", cols="<<matrix.total_cols()<<", nnz="<<matrix.total_nnz()<<"\n";

      size_t rowIdx = 0;
      size_t nxtRowIdx = 0;

      for(size_t ii = 0; ii < matrix.total_rows(); ii++)
      {
         rowIdx = matrix.m_rowPtr[ii];
         nxtRowIdx = matrix.m_rowPtr[ii+1];

         for(size_t jj=rowIdx; jj<nxtRowIdx; jj++)
         {
            os<< "row: "<<std::setw(8)<<ii<<", col: "<<std::setw(8)<<matrix.m_colInd[jj]<<", value"<<std::setw(12)<<matrix.m_values[jj]<<"\n";
         }
      }

      os<<"\n";
      return os;
   }


public:

   /*!
    * Returns number of non zero elements in the SparseMatrix.
    * \return count of non zero elements of the SparseMatrix.
    */
   size_t total_nnz() const
   {
      return m_nnz;
   }

   /*!
    * Returns total number of rows in the SparseMatrix.
    * \return rows in the SparseMatrix.
    */
   size_t total_rows() const
   {
      return m_rows;
   }

   /*!
    * Returns total number of columns in the SparseMatrix.
    * \return columns in the SparseMatrix.
    */
   size_t total_cols() const
   {
      return m_cols;
   }

   /*!
    * Returns pointer to actual non zero values in the SparseMatrix.
    * \return pointer to actual non zero values in the SparseMatrix.
    */
   T* get_values() const
   {
      return m_values;
   }

   size_t* get_row_pointers() const
   {
      return m_rowPtr;
   }

   size_t* get_col_indices() const
   {
      return m_colInd;
   }

   size_t get_rowSize(size_t row) const
   {
      return (m_rowPtr[row+1]-m_rowPtr[row]);
   }

   size_t get_rowOffsetFromStart(size_t row) const
   {
      if(row<total_rows())
         return m_rowPtr[row];

      return m_nnz;
   }
   
   SparseMat<T> hostProxy()
   {
      SparseMat<T> proxy;
      proxy.data = this->m_values;
      proxy.row_offsets = this->m_rowPtr;
      proxy.col_indices = this->m_colInd;
      proxy.count = this->m_nnz;
      return proxy;
   }

public:

   iterator begin(unsigned row = 0);

   // Element access using 'at' operator
   T at(size_t row, size_t col ) const;
   const T& at(size_t index) const;

   // Similar to 'at' operator but don't do memory check etc. must faster used internally in implementation.
   const T& at_internal(size_t row, size_t col ) const;
   T& at_internal(size_t row, size_t col );
   const T& at_internal(size_t index) const;

   // Element access using () operator
   const T& operator()(const size_t row, const size_t col) const;
   T operator()(const size_t row, const size_t col);

   // will resize the matrix, can be dangerous, used with care
   void resize(SparseMatrix<T> &copy, bool retainData);

   void updateHost(bool = true) const;
   void invalidateDeviceData(bool = true);
   void updateHostAndInvalidateDevice();

#ifdef SKEPU_OPENCL
   device_pointer_type_cl updateDevice_CL(T* start, size_t elems, backend::Device_CL* device, bool copy);
   device_pointer_index_type_cl updateDevice_Index_CL(size_t* start, size_t elems, backend::Device_CL* device, bool copy);
   void flush_CL();
#endif

#ifdef SKEPU_CUDA
   device_pointer_type_cu updateDevice_CU(T* start, size_t elems, unsigned int deviceID, bool copy);
   device_pointer_index_type_cu updateDevice_Index_CU(size_t* start, size_t elems, unsigned int deviceID, bool copy);
   void flush_CU();
   bool isSparseMatrixOnDevice_CU(unsigned int deviceID);
   bool isModified_CU(unsigned int deviceID);
#endif
  

private:

#ifdef SKEPU_OPENCL
   void updateHost_CL() const;
   void invalidateDeviceData_CL();
   void releaseDeviceAllocations_CL();
#endif

#ifdef SKEPU_CUDA
   void updateHost_CU() const;
   void invalidateDeviceData_CU();
   void releaseDeviceAllocations_CU();
#endif

#ifdef SKEPU_CUDA
   std::map<std::pair< unsigned int, std::pair< T*, size_t > >, device_pointer_type_cu > m_deviceMemPointers_CU;
   std::map<std::pair< unsigned int, std::pair< size_t*, size_t > >, device_pointer_index_type_cu > m_deviceMemIndexPointers_CU;
#endif

#ifdef SKEPU_OPENCL
   std::map<std::pair< cl_device_id, std::pair< T*, size_t > >, device_pointer_type_cl > m_deviceMemPointers_CL;
   std::map<std::pair< cl_device_id, std::pair< size_t*, size_t > >, device_pointer_index_type_cl > m_deviceMemIndexPointers_CL;
#endif

}; // end class SparseMatrix


/*!
 *  \class SparseMatrix::iterator
 *
 *  \brief An sparse matrix iterator class that tranverses row-wise.
 *
 *  An iterator class for \p skepu::SparseMatrix. It traverses a SparseMatrix elements range
 *  which is \p skepu::Matrix default style. It behaves like the 1D container iterators like iterator for \p std::vector
 *  but similar to \p skepu::Matrix it sometimes returns a \p proxy_elem instead of the actual
 *  element. Also makes sure the matrix is properly synchronized with device before returning
 *  any elements.
 */
template <typename T>
class SparseMatrix<T>::iterator
{

public: //-- Constructors & Destructor --//
   iterator(SparseMatrix<T> *parent, T *start, T *end);


public: //-- Extras --//

//   Index2D getIndex() const;

   SparseMatrix<T>& getParent() const
   {
      return *this->m_parent;   
   }
   
   T* getAddress() const;

   //Does not care about device data, use with care
   T& operator()(const ssize_t index);

   //Does care about device data, uses updateDevice, for readonly access
   const T& operator()(const ssize_t index) const;

public: //-- Operators --//

   //Does care about device data, uses updateDevice, for readwrite access
   T& operator[](const ssize_t index);

   //Does care about device data, uses updateDevice, for readonly access
   const T& operator[](const ssize_t index) const;
   
   
   
   const typename SparseMatrix<T>::iterator& operator++(); //Prefix
   
   typename SparseMatrix<T>::difference_type operator-(const iterator& i) const;
   
   const T& operator*() const;

   size_t size()
   {
      return m_size;
   }

   SparseMat<T> hostProxy()
   {
      SparseMat<T> proxy;
      proxy.data = this->m_parent->m_values;
      proxy.row_offsets = this->m_parent->m_rowPtr;
      proxy.col_indices = this->m_parent->m_colInd;
      proxy.count = this->m_parent->m_nnz;
      return proxy;
   }

private: //-- Data --//
   size_t m_size;
   SparseMatrix<T>* m_parent;
   T *m_start;
   T *m_end;
};



} // end namespace skepu2

#include "backend/impl/sparse_matrix/sparse_matrix_iterator.inl"
#include "backend/impl/sparse_matrix/sparse_matrix.inl"
#include "backend/impl/sparse_matrix/sparse_matrix_cl.inl"
#include "backend/impl/sparse_matrix/sparse_matrix_cu.inl"

#endif  /* _MATRIX_H */

