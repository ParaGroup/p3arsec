/*! \file helper_methods.h
 *  \brief Contains few helper methods that are used globally by different classes.
 */

/*!
 *  \ingroup helpers
 */

#ifndef _HELPER_METHODS_H
#define _HELPER_METHODS_H

#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>

namespace skepu2
{
	namespace backend
	{
		/*!
		 * Method to remove leading and trailing spaces from a string.
		 */
		static const std::string trimSpaces(const std::string& pString, const std::string& pWhitespace = " \t")
		{
			const size_t beginStr = pString.find_first_not_of(pWhitespace);
			if (beginStr == std::string::npos)
				return "";
			
			const size_t endStr = pString.find_last_not_of(pWhitespace);
			const size_t range = endStr - beginStr + 1;
			
			return pString.substr(beginStr, range);
		}
		
		
		/*!
		 * Method to get a random number between a given range.
		 */
		template <typename T>
		inline T get_random_number(T min, T max)
		{
			return (T)(rand() % (int)(max-min+1) + min);
		}
		
		
		/*!
		 * Method to allocate host memory of a given size. Can do pinned memory allocation if enabled.
		 */
		template<typename T>
		void allocateHostMemory(T* &data, const size_t numElems)
		{
#if defined(SKEPU_CUDA) && defined(USE_PINNED_MEMORY)
			cudaError_t status = cudaMallocHost((void**)&data, numElems*sizeof(T));
			if (status != cudaSuccess)
			{
				SKEPU_ERROR("Error allocating pinned host memory\n");
			}
#else
			data = new T[numElems];
			if (!data)
				SKEPU_ERROR("Memory allocation failed\n");
#endif   
		}
		
		
		/*!
		 * Method to deallocate host memory.
		 */
		template<typename T>
		void deallocateHostMemory(T *data)
		{
			if (!data)
				return;
			   
#if defined(SKEPU_CUDA) && defined(USE_PINNED_MEMORY)
			cudaError_t status = cudaFreeHost(data);
			if (status != cudaSuccess)
				SKEPU_ERROR("Error de-allocating pinned host memory.\n");
#else
			delete[] data;
#endif
		}
		
	} // end namespace backend
} // end namespace skepu2

#endif
