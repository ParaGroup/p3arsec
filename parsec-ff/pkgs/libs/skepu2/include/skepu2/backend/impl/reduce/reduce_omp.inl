/*! \file reduce_omp.inl
*  \brief Contains the definitions of OpenMP specific member functions for the Reduce skeleton.
 */

#ifdef SKEPU_OPENMP

#include <omp.h>

namespace skepu2
{
	namespace backend
	{
		/*!
		 *  Performs the Reduction on a whole Matrix. Returns a \em SkePU vector of reduction result.
		 *  Using \em OpenMP as backend.
		 */
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		Vector<typename ReduceFunc::Ret> &Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::OMP(Vector<T> &res, Matrix<T>& arg)
		{
			const size_t rows = arg.total_rows();
			const size_t cols = arg.total_cols();
			const size_t size = rows * cols;
			
			DEBUG_TEXT_LEVEL1("OpenMP Reduce (Matrix 1D): rows = " << rows << ", cols = " << cols << "\n");
			
			// Make sure we are properly synched with device data
			arg.updateHost();
			
			Vector<T> result(rows);
			T *data = arg.getAddress();
			
			// Set up thread indexing
			omp_set_num_threads(std::min(this->m_selected_spec->CPUThreads(), size / 2));
			const size_t nthr = omp_get_max_threads();
			const size_t q = size / nthr;
			const size_t rest = size % nthr;
			
			// Now, we safely assume that there are at least as number of rows to process as #threads available
			// schedule rows to each thread
			const size_t rowsPerThread = rows / nthr;
			const size_t restRows = rows % nthr;
			
			// we divide the "N" remainder rows to first "N" threads instead of giving it to last thread to achieve better load balancing
#pragma omp parallel
			{
				const size_t myid = omp_get_thread_num();
				size_t firstRow = myid * rowsPerThread;
				size_t lastRow;
				
				if(myid!=0)
					firstRow += (myid<restRows)? myid:restRows;
				
				if(myid < restRows)
					lastRow = firstRow+rowsPerThread+1;
				else
					lastRow = firstRow+rowsPerThread;
				
				for (size_t r = firstRow; r < lastRow; ++r)
				{
					size_t base = r*cols;
					T psum = data[base];
					for(size_t c=1; c<cols; ++c)
					{
						psum = ReduceFunc::OMP(psum, data[base+c]);
					}
					res(r) = psum;
				}
			}
			
			return res;
		}
		
		
		/*!
		 *  Performs the Reduction on a range of elements. Returns a scalar result. Divides the elements among all
		 *  \em OpenMP threads and does reduction of the parts in parallel. The results from each thread are then
		 *  reduced on the CPU.
		 */
		template<typename ReduceFunc, typename CUDAKernel, typename CLKernel>
		template<typename Iterator>
		typename ReduceFunc::Ret Reduce1D<ReduceFunc, CUDAKernel, CLKernel>
		::OMP(size_t size, T &res, Iterator arg)
		{
			// Make sure we are properly synched with device data
			arg.getParent().updateHost();
			
			// Set up thread indexing
			omp_set_num_threads(std::min(this->m_selected_spec->CPUThreads(), size / 2));
			const size_t nthr = omp_get_max_threads();
			const size_t q = size / nthr;
			const size_t rest = size % nthr;
			
			std::vector<T> parsums(nthr);
			
#pragma omp parallel
			{
				const size_t myid = omp_get_thread_num();
				const size_t first = myid * q;
				const size_t last = (myid+1) * q + ((myid == nthr-1) ? rest : 0);
				
				T psum = arg(first);
				for (size_t i = first+1; i < last; ++i)
					psum = ReduceFunc::OMP(psum, arg(i));
				parsums[myid] = psum;
			}
			
			for (auto it = parsums.begin(); it != parsums.end(); ++it)
				res = ReduceFunc::OMP(res, *it);
			
			return res;
		}
		
		
		/*!
		 *  Performs the 2D Reduction (First row-wise then column-wise) on a
		 *  input Matrix. Returns a scalar result.
		 *  Using the \em OpenMP as backend.
		 */
		template<typename ReduceFuncRowWise, typename ReduceFuncColWise, typename CUDARowWise, typename CUDAColWise, typename CLKernel>
		typename ReduceFuncRowWise::Ret Reduce2D<ReduceFuncRowWise, ReduceFuncColWise, CUDARowWise, CUDAColWise, CLKernel>
		::OMP(T &res, Matrix<T>& arg)
		{
			// Make sure we are properly synched with device data
			arg.updateHost();
			
			const size_t rows = arg.total_rows();
			const size_t cols = arg.total_cols();
			const size_t size = rows * cols;
			
			// Set up thread indexing
			omp_set_num_threads(std::min(this->m_selected_spec->CPUThreads(), size / 2));
			const size_t nthr = omp_get_max_threads();
			const size_t q = size / nthr;
			const size_t rest = size % nthr;
			
			// Now, we safely assume that there are at least as number of rows to process as #threads available
			// schedule rows to each thread
			const size_t rowsPerThread = rows / nthr;
			const size_t restRows = rows % nthr;
			std::vector<T> parsums(rows);
			T *data = arg.getAddress();
			
			// we divide the "N" remainder rows to first "N" threads instead of giving it to last thread to achieve better load balancing
#pragma omp parallel
			{
				const size_t myid = omp_get_thread_num();
				size_t firstRow = myid * rowsPerThread;
				size_t lastRow;
				
				if (myid != 0)
					firstRow += (myid<restRows) ? myid : restRows;
				
				if (myid < restRows)
					lastRow = firstRow+rowsPerThread+1;
				else
					lastRow = firstRow+rowsPerThread;
				
				for (size_t r = firstRow; r < lastRow; ++r)
				{
					const size_t base = r * cols;
					T psum = data[base];
					for (size_t c = 1;  c < cols; ++c)
						psum = ReduceFuncRowWise::OMP(psum, data[base+c]);
					parsums[r] = psum;
				}
			}
			
			if (rows / nthr > 8) // if sufficient work to do it in parallel
				ompVectorReduce(res, parsums, nthr);
			else
			{
				// do it sequentially
				for (auto it = parsums.begin(); it != parsums.end(); ++it)
					res = ReduceFuncColWise::OMP(res, *it);
			}
			
			return res;
		}
		
		
		/*!
		 *  A helper provate method used to do final 1D reduction. Used internally.
		 */
		template<typename ReduceFuncRowWise, typename ReduceFuncColWise, typename CUDARowWise, typename CUDAColWise, typename CLKernel>
		typename ReduceFuncRowWise::Ret Reduce2D<ReduceFuncRowWise, ReduceFuncColWise, CUDARowWise, CUDAColWise, CLKernel>
		::ompVectorReduce(T &res, std::vector<T> &input, size_t numThreads)
		{
			// Set up thread indexing
			const size_t size = input.size();
			omp_set_num_threads(std::min(numThreads, size / 2));
			const size_t nthr = omp_get_max_threads();
			const size_t q = size / nthr;
			const size_t rest = size % nthr;
			
			std::vector<T> parsums(nthr);
			
#pragma omp parallel
			{
				const size_t myid = omp_get_thread_num();
				const size_t first = myid*q;
				const size_t last = (myid+1) * q + ((myid == nthr-1) ? rest : 0);
				
				T psum = input[first];
				for (size_t i = first+1; i < last; ++i)
					psum = ReduceFuncColWise::OMP(psum, input[i]);
				parsums[myid] = psum;
			}
			
			for (auto it = parsums.begin(); it != parsums.end(); ++it)
				res = ReduceFuncColWise::OMP(res, *it);
			
			return res;
		}
		
	} // end namespace backend
} // end namespace skepu2

#endif

