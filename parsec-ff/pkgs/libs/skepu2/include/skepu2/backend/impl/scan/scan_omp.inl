/*! \file scan_cpu.inl
 *  \brief Contains the definitions of CPU specific member functions for the Scan skeleton.
 */

#ifdef SKEPU_OPENMP

namespace skepu2
{
	namespace backend
	{
		template<typename ScanFunc, typename CUDAScan, typename CUDAScanUpdate, typename CUDAScanAdd, typename CLKernel>
		template<typename OutIterator, typename InIterator>
		void Scan<ScanFunc, CUDAScan, CUDAScanUpdate, CUDAScanAdd, CLKernel>
		::OMP(size_t size, OutIterator res, InIterator arg, ScanMode mode, T initial)
		{
			// Make sure we are properly synched with device data
			res.getParent().invalidateDeviceData();
			arg.getParent().updateHost();
			
			// Setup parameters needed to parallelize with OpenMP
			omp_set_num_threads(std::min(this->m_selected_spec->CPUThreads(), size / 2));
			const size_t nthr = omp_get_max_threads();
			const size_t q = size / nthr;
			const size_t rest = size % nthr;
			
			// Array to store partial thread results in.
			std::vector<T> offset_array(nthr);
			
			// Process first element here
			*res = (mode == ScanMode::Inclusive) ? *arg++ : initial;
			
#pragma omp parallel
			{
				const size_t myid = omp_get_thread_num();
				const size_t first = myid * q;
				const size_t last = (myid + 1) * q + ((myid == nthr - 1) ? rest : 0);
				
				// First let each thread make their own scan and saved the result in a partial result array.
				if (myid != 0) res(first) = arg(first-1);
				for (size_t i = first + 1; i < last; ++i)
				{
					res(i) = ScanFunc::OMP(res(i-1), arg(i-1));
				}
				offset_array[myid] = res(last-1);
				
#pragma omp barrier
				
				// Let the master thread scan the partial result array
#pragma omp master
				for (size_t i = 1; i < nthr; ++i)
				{
					offset_array[i] = ScanFunc::OMP(offset_array[i-1], offset_array[i]);
				}
				
#pragma omp barrier
				
				if (myid != 0)
				{
					// Add the scanned partial results to each threads work batch.
					for (size_t i = first; i < last; ++i)
					{
						res(i) = ScanFunc::OMP(res(i), offset_array[myid-1]);
					}
				}
			}
		}
		
	}
}

#endif // SKEPU_OPENMP
