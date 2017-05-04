/*! \file device_mem_pointer_cu.h
*  \brief Contains a class declaration for an object which represents an CUDA device memory allocation for Vector container.
 */

#ifndef DEVICE_MEM_POINTER_CU_H
#define DEVICE_MEM_POINTER_CU_H

#ifdef SKEPU_CUDA

#include <iostream>
#include <cuda.h>


#include "device_cu.h"
#include "mem_pointer_base.h"

namespace skepu2
{
	namespace backend
	{

		/*!
		*  \ingroup helpers
		*/   
		/*!
		* The structure that will be used to keep information about what parts need to be copied from where
		*/
		template<typename T>
		struct UpdateInf
		{
			int srcDevId;
			T* src;
			int srcOffset;
			int dstOffset;
			size_t copySize;
			bool srcIsHost; /*! true if src is host else false ... */
		};

/*! specified the maximum size for number of sub ranges that can be tracked for a given copy... */
#define MAX_RANGES 10

		/*! U.Dastgeer: specified the maximum size for total sub copies needed to fill a copy data...
		* C.Parisi: originally 10, raised to 33 for concurrent kernel execution on devices up to Compute Capability 5.2 
*/
#define MAX_COPYINF_SIZE 33

		/*!
		*  \ingroup helpers
		*/

		/*!
		*  \class DeviceMemPointer_CU
		*
		*  \brief A class representing a CUDA device memory allocation for container.
		*
		*  This class represents a CUDA device 1D memory allocation and controls the data transfers between
		*  host and device.
		*/
		template <typename T>
		class DeviceMemPointer_CU : public MemPointerBase
		{

		public:
			DeviceMemPointer_CU(T* start, size_t numElements, Device_CU *device, std::string name="");
			DeviceMemPointer_CU(T* start, size_t rows, size_t cols, Device_CU *device, bool usePitch=false, std::string name="");

//       DeviceMemPointer_CU(T* root, T* start, size_t numElements, Device_CU *device);
			~DeviceMemPointer_CU();

			void copyHostToDevice(size_t numElements = 0) const;
			void copyDeviceToHost(size_t numElements = 0) const;

			void copiesOverlapInf(DeviceMemPointer_CU<T> *otherCopy, UpdateInf<T>* updateStruct, size_t &sizeUpdStr);
	
			size_t getMemSize();
			void clearDevicePointer();

			T* getDeviceDataPointer() const;
			unsigned int getDeviceID() const;

			void changeDeviceData(bool condition = true);
			bool deviceDataHasChanged() const;

			void markCopyInvalid();
			bool isCopyValid() const;
   
// #if SKEPU_DEBUG>0   
			std::string m_nameVerbose;
// #endif   

			size_t m_pitch;

			size_t m_rows;

			size_t m_cols;

			bool doCopiesOverlap(DeviceMemPointer_CU<T> *otherCopy, bool oneUnitCheck = false);
			bool doOverlapAndCoverFully(DeviceMemPointer_CU<T> *otherCopy);
			bool doRangeOverlap(T *hostPtr, size_t numElements);
			void copyInfFromHostToDevice(UpdateInf<T>* updateStruct, size_t &sizeUpdStr);
			void copyAllRangesToDevice(UpdateInf<T>* updateStruct, const size_t sizeUpdStr, size_t streamID = 0);

			std::pair<T*, int> m_rangesToCompare[MAX_RANGES];
			size_t m_numOfRanges;

			T* m_hostDataPointer;
			T* m_deviceDataPointer;
			size_t m_numElements;

			void resetRanges()
			{
      /*! ranges that should be checked for overlap with other copies */
				m_numOfRanges = 0;
				m_rangesToCompare[m_numOfRanges] = std::make_pair(m_hostDataPointer, m_numElements);
				m_numOfRanges++;
			}

		private:
			unsigned int m_deviceID;
			Device_CU *m_dev;

			bool m_usePitch;

			/*!
			* marks that the copy contents are not valid...
			* the copy could be used for write purpose but any read of this copy would demand copying data first
			*/
			mutable bool m_valid;

			mutable bool m_deviceDataHasChanged;

			bool freeUpDeviceMem();
		};

		/*!
		* Checks whether the copy passed as argument has a subset of elements range to the one that object points to.
		*/
		template <typename T>
		bool DeviceMemPointer_CU<T>::doOverlapAndCoverFully(DeviceMemPointer_CU<T> *otherCopy)
		{
			if(m_hostDataPointer <= otherCopy->m_hostDataPointer && (m_hostDataPointer+m_numElements) >= (otherCopy->m_hostDataPointer+otherCopy->m_numElements))
				return true;

			return false;
		}

		/*!
		* it returns true if there exist any range (needs to be written) that
		* is overlapping to the otherCopy
		*/
		template <typename T>
		bool DeviceMemPointer_CU<T>::doCopiesOverlap(DeviceMemPointer_CU<T> *otherCopy, bool oneUnitCheck)
		{
			if(oneUnitCheck)
				assert(m_numOfRanges == 1 && otherCopy->m_numOfRanges == 1);
   
			if(m_numOfRanges < 1)
				return false;

			for(size_t i=0; i<m_numOfRanges; ++i)
			{
				T *hostDataPointer = m_rangesToCompare[i].first;
				size_t numElements = m_rangesToCompare[i].second;

				if( hostDataPointer >= otherCopy->m_hostDataPointer && hostDataPointer < (otherCopy->m_hostDataPointer + otherCopy->m_numElements) )
					return true;

				if( otherCopy->m_hostDataPointer >= hostDataPointer && otherCopy->m_hostDataPointer < (hostDataPointer + numElements) )
					return true;
			}
			return false;
		}

		/*!
		* Checks whether there exists some overlap between elements range covered by current copy to the one passed as argument.
		*/
		template <typename T>
		bool DeviceMemPointer_CU<T>::doRangeOverlap(T *hostDataPointer, size_t numElements)
		{
			if( (m_hostDataPointer+m_numElements) <= hostDataPointer )
				return false;

			if( (hostDataPointer+numElements) <= m_hostDataPointer )
				return false;

			return true;
		}

		/*!
		*  The constructor allocates a certain amount of space in device memory and stores a pointer to
		*  some data in host memory.
		*
		*  \param start Pointer to data in host memory.
		*  \param numElements Number of elements to allocate memory for.
		*  \param device pointer to Device_CU object of a valid CUDA device to allocate memory on.
		*/
		template <typename T>
		DeviceMemPointer_CU<T>::DeviceMemPointer_CU(T* start, size_t numElements, Device_CU *device, std::string name) : m_hostDataPointer(start), m_numElements(numElements), m_dev(device), m_rows(1), m_cols(numElements), m_pitch(numElements), m_valid(false), m_usePitch(false), m_numOfRanges(0)
		{
			m_nameVerbose = name;
   
			size_t sizeVec = m_numElements*sizeof(T);

			m_deviceID = m_dev->getDeviceID();
			CHECK_CUDA_ERROR(cudaSetDevice(m_deviceID));
   
			DEBUG_TEXT_LEVEL1(m_nameVerbose + " Alloc: " <<m_numElements << ", GPU_" << m_deviceID << "\n")

				cudaError_t er;
			DeviceAllocations_CU<int>* dev_alloc = DeviceAllocations_CU<int>::getInstance();
			er  = cudaMalloc((void**)&m_deviceDataPointer, sizeVec);
			if (er == cudaErrorMemoryAllocation)
			{
				freeUpDeviceMem();
			}
			else
				CHECK_CUDA_ERROR(er);
			// add to list of device allocations
			dev_alloc->addAllocation((void*)m_deviceDataPointer,this,m_deviceID);

			/*! ranges that should be checked for overlap with other copies */
			m_rangesToCompare[m_numOfRanges] = std::make_pair(m_hostDataPointer, m_numElements);
			m_numOfRanges++;

			m_deviceDataHasChanged = false;
		}



		/*!
		*  The constructor allocates a certain amount of space in device memory and stores a pointer to
		*  some data in host memory.
		*
		*  \param start Pointer to data in host memory.
		*  \param rows Number of rows to allocate memory for.
		*  \param cols Number of columns to allocate memory for.
		*  \param device pointer to Device_CU object of a valid CUDA device to allocate memory on.
		*  \param usePitch To specify whether to use padding to ensure proper coalescing for row-wise access from CUDA global memory.
		*/
		template <typename T>
		DeviceMemPointer_CU<T>::DeviceMemPointer_CU(T* start, size_t rows, size_t cols, Device_CU *device, bool usePitch, std::string name) : m_hostDataPointer(start), m_numElements(rows*cols), m_rows(rows), m_cols(cols), m_dev(device), m_valid(false), m_usePitch(usePitch), m_numOfRanges(0)
		{
			m_nameVerbose = name;
   
			size_t sizeVec = m_numElements*sizeof(T);

			m_deviceID = m_dev->getDeviceID();
			CHECK_CUDA_ERROR(cudaSetDevice(m_deviceID));
   
			DEBUG_TEXT_LEVEL1(m_nameVerbose + " Alloc: " <<m_numElements << ", GPU_" << m_deviceID << "\n")

				if(m_usePitch)
			{
				CHECK_CUDA_ERROR(cudaMallocPitch((void**)&m_deviceDataPointer, &m_pitch, cols * sizeof(T), rows));
				m_pitch = (m_pitch)/sizeof(T);
			}
			else
			{
				cudaError_t er;
				DeviceAllocations_CU<int>* dev_alloc = DeviceAllocations_CU<int>::getInstance();
				er  = cudaMalloc((void**)&m_deviceDataPointer, sizeVec);
				if (er == cudaErrorMemoryAllocation)
				{
					freeUpDeviceMem();
				}
				else
					CHECK_CUDA_ERROR(er);
 			// add to list of device allocations
				dev_alloc->addAllocation((void*)m_deviceDataPointer,this,m_deviceID);
				m_pitch = cols;
			}
			
			/*! ranges that should be checked for overlap with other copies */
			m_rangesToCompare[m_numOfRanges] = std::make_pair(m_hostDataPointer, m_numElements);
			m_numOfRanges++;
			m_deviceDataHasChanged = false;
		}



		/*!
		*  The destructor releases the allocated device memory.
		*/
		template <typename T>
		DeviceMemPointer_CU<T>::~DeviceMemPointer_CU()
		{
			DEBUG_TEXT_LEVEL1(m_nameVerbose + " DeAlloc: " <<m_numElements <<", GPU_" << m_deviceID << "\n")

				CHECK_CUDA_ERROR(cudaSetDevice(m_deviceID));

			DeviceAllocations_CU<int>::getInstance()->removeAllocation(m_deviceDataPointer,this,m_deviceID);	
	
			cudaFree(m_deviceDataPointer);
			m_deviceDataPointer = NULL;
		}

		/*!
		* copies data from host to device for remaining portions of the copy
		* assumes that host copy is valid...
		* \param updateStruct the array of structures that keep track of what needs to be copied
		* \param sizeUpdStr the length of \p updateStruct array
		*/
		template <typename T>
		void DeviceMemPointer_CU<T>::copyInfFromHostToDevice(UpdateInf<T>* updateStruct, size_t &sizeUpdStr)
		{
			for(size_t i=0; i<m_numOfRanges; ++i)
			{
      /*! get pointer and size information for a range that is not copied yet */
				T *hostDataPointer = m_rangesToCompare[i].first;
				size_t numElements = m_rangesToCompare[i].second;

				assert((hostDataPointer - m_hostDataPointer)>=0);
				int offset = hostDataPointer - m_hostDataPointer;

				size_t sizeVec = numElements*sizeof(T);

				updateStruct[sizeUpdStr].srcDevId = -1;
				updateStruct[sizeUpdStr].src = m_hostDataPointer;
//          updateStruct[sizeUpdStr].dst = this;
				updateStruct[sizeUpdStr].srcOffset = offset;
				updateStruct[sizeUpdStr].dstOffset = offset;
				updateStruct[sizeUpdStr].copySize = sizeVec;
				updateStruct[sizeUpdStr].srcIsHost = true;
				sizeUpdStr++;
				assert(sizeUpdStr < MAX_COPYINF_SIZE);

				/*! delete this configuration... */
				if(i!=m_numOfRanges-1) // if not last then shift elements back...
				{
					for(size_t j=i+1; j<m_numOfRanges; ++j)
					{
						m_rangesToCompare[j-1] = m_rangesToCompare[j];
					}
				}
				--i;
				--m_numOfRanges; // decremement total size
			}
		}


		/*!
		* Finds out what (and how much) elements needs be copied in the current device copy from the passed copy.
		* If the passed copy is a superset then all required elements can be copied otherwise some ranges may need to 
		* be copied from other sources...
		* \param otherCopy The other device copy
		* \param updateStruct The array of structure which is updated with new entries regarding data copy information
		* \param sizeUpdStr the length of \p updateStruct array
		*/
		template <typename T>
		void DeviceMemPointer_CU<T>::copiesOverlapInf(DeviceMemPointer_CU<T> *otherCopy, UpdateInf<T>* updateStruct, size_t &sizeUpdStr)
		{
			for(size_t i=0; i<m_numOfRanges; ++i)
			{
				/*! get pointer and size information for a range that is not copied yet */
				T *hostDataPointer = m_rangesToCompare[i].first;
				size_t numElements = m_rangesToCompare[i].second;

				/*! first check whether this range that is still not copied overlaps with this potential source copy, i.e., otherCopy, if not skip this range for further processing */
				if(otherCopy->doRangeOverlap(hostDataPointer, numElements) == false)
					continue;

				assert((hostDataPointer - m_hostDataPointer)>=0);
				int offset = hostDataPointer - m_hostDataPointer;
      

				/*! Scenario 1: otherCopy covers the whole what needs to copied... no need to copy anything from host.... */
				if( hostDataPointer >= otherCopy->m_hostDataPointer && (hostDataPointer + numElements) <= (otherCopy->m_hostDataPointer + otherCopy->m_numElements) )
				{
					size_t sizeVec = numElements*sizeof(T);
					int srcoffset = hostDataPointer - otherCopy->m_hostDataPointer;

					updateStruct[sizeUpdStr].srcDevId = otherCopy->m_deviceID;
					updateStruct[sizeUpdStr].src = otherCopy->m_deviceDataPointer;
//             updateStruct[sizeUpdStr].dst = this;
					updateStruct[sizeUpdStr].srcOffset = srcoffset;
					updateStruct[sizeUpdStr].dstOffset = offset;
					updateStruct[sizeUpdStr].copySize = sizeVec;
					updateStruct[sizeUpdStr].srcIsHost = false;
					sizeUpdStr++;
					assert(sizeUpdStr < MAX_COPYINF_SIZE);

					/*! delete this configuration... */
					if(i!=m_numOfRanges-1) // if not last then shift elements back...
					{
						for(int j=i+1; j<m_numOfRanges; ++j)
						{
							m_rangesToCompare[j-1] = m_rangesToCompare[j];
						}
					}
					--i;
					--m_numOfRanges; // decremement total size
				}
				/*! Scenario 2: otherCopy is fully nested from both sides... need to copy from some other source both to left and to the right.... */
				else if( otherCopy->m_hostDataPointer >= hostDataPointer && (otherCopy->m_hostDataPointer + otherCopy->m_numElements) <= (hostDataPointer + numElements) )
				{
					size_t sizeVec = otherCopy->m_numElements*sizeof(T);
					int dstoffset = otherCopy->m_hostDataPointer - hostDataPointer;

					updateStruct[sizeUpdStr].srcDevId = otherCopy->m_deviceID;
					updateStruct[sizeUpdStr].src = otherCopy->m_deviceDataPointer;
//             updateStruct[sizeUpdStr].dst = this;
					updateStruct[sizeUpdStr].srcOffset = 0;
					updateStruct[sizeUpdStr].dstOffset = offset + dstoffset;
					updateStruct[sizeUpdStr].copySize = sizeVec;
					updateStruct[sizeUpdStr].srcIsHost = false;
					sizeUpdStr++;
					assert(sizeUpdStr < MAX_COPYINF_SIZE);

					/*! delete this configuration... */
					if(i!=m_numOfRanges-1) // if not last then shift elements back...
					{
						for(size_t j=i+1; j<m_numOfRanges; ++j)
						{
							m_rangesToCompare[j-1] = m_rangesToCompare[j];
						}
					}
					--i;
					--m_numOfRanges; // decremement total size

					/*! add configuration for left part */
					if(dstoffset>0) // its posisble that the dstoffset is 0
					{
						m_rangesToCompare[m_numOfRanges] = std::make_pair(hostDataPointer, dstoffset);
						m_numOfRanges++;
						assert(m_numOfRanges<MAX_RANGES);
					}

					/*! add configuration for right part */
					int size = ( (hostDataPointer+numElements) - (otherCopy->m_hostDataPointer + otherCopy->m_numElements) );
					if(size>0)
					{
						int tmpoffset = (otherCopy->m_hostDataPointer + otherCopy->m_numElements) - hostDataPointer;
						m_rangesToCompare[m_numOfRanges] = std::make_pair(hostDataPointer+tmpoffset, size);
						m_numOfRanges++;
						assert(m_numOfRanges<MAX_RANGES);
					}
				}
				/*! Scenario 3: otherCopy is partially nested from left side... need to copy from host to left.... */
				else if( otherCopy->m_hostDataPointer >= hostDataPointer && (otherCopy->m_hostDataPointer) < (hostDataPointer + numElements) )
				{
					size_t sizeVec = ( (hostDataPointer + numElements) -  otherCopy->m_hostDataPointer) * sizeof(T);
					int dstoffset = otherCopy->m_hostDataPointer - hostDataPointer;
					
					updateStruct[sizeUpdStr].srcDevId = otherCopy->m_deviceID;
					updateStruct[sizeUpdStr].src = otherCopy->m_deviceDataPointer;
//             updateStruct[sizeUpdStr].dst = this;
					updateStruct[sizeUpdStr].srcOffset = 0;
					updateStruct[sizeUpdStr].dstOffset = offset + dstoffset;
					updateStruct[sizeUpdStr].copySize = sizeVec;
					updateStruct[sizeUpdStr].srcIsHost = false;
					sizeUpdStr++;
					assert(sizeUpdStr < MAX_COPYINF_SIZE);
					
					/*! delete this configuration... */
					if(i!=m_numOfRanges-1) // if not last then shift elements back...
					{
						for(size_t j=i+1; j<m_numOfRanges; ++j)
						{
							m_rangesToCompare[j-1] = m_rangesToCompare[j];
						}
					}
					--i;
					--m_numOfRanges; // decremement total size
					
					/*! add configuration for left part */
					if(dstoffset>0) // its posisble that the dstoffset is 0
					{
						m_rangesToCompare[m_numOfRanges] = std::make_pair(hostDataPointer, dstoffset);
						m_numOfRanges++;
						assert(m_numOfRanges<MAX_RANGES);
					}
				}
				/*! Scenario 4: otherCopy is partially nested from right side... need to copy from host to right.... */
				else if( hostDataPointer >= otherCopy->m_hostDataPointer && hostDataPointer < (otherCopy->m_hostDataPointer + otherCopy->m_numElements) )
				{
					size_t sizeVec = ( (otherCopy->m_hostDataPointer + otherCopy->m_numElements) -  hostDataPointer) * sizeof(T);
					int srcoffset = hostDataPointer - otherCopy->m_hostDataPointer;

					updateStruct[sizeUpdStr].srcDevId = otherCopy->m_deviceID;
					updateStruct[sizeUpdStr].src = otherCopy->m_deviceDataPointer;
//             updateStruct[sizeUpdStr].dst = this;
					updateStruct[sizeUpdStr].srcOffset = srcoffset;
					updateStruct[sizeUpdStr].dstOffset = offset;
					updateStruct[sizeUpdStr].copySize = sizeVec;
					updateStruct[sizeUpdStr].srcIsHost = false;
					sizeUpdStr++;
					assert(sizeUpdStr < MAX_COPYINF_SIZE);

         /*! delete this configuration... */
					if(i!=m_numOfRanges-1) // if not last then shift elements back...
					{
						for(size_t j=i+1; j<m_numOfRanges; ++j)
						{
							m_rangesToCompare[j-1] = m_rangesToCompare[j];
						}
					}
					--i;
					--m_numOfRanges; // decremement total size

         /*! add configuration for right part */
					int size = ( (hostDataPointer+numElements) - (otherCopy->m_hostDataPointer + otherCopy->m_numElements) );
					if(size > 0)
					{
						int tmpoffset = (otherCopy->m_hostDataPointer + otherCopy->m_numElements) - hostDataPointer;
						m_rangesToCompare[m_numOfRanges] = std::make_pair(hostDataPointer+tmpoffset, size);
						m_numOfRanges++;
						assert(m_numOfRanges<MAX_RANGES);
					}
				}
				else
					assert(false);
			}
		}
		
		template <typename T>
		size_t DeviceMemPointer_CU<T>::getMemSize()
		{
			return sizeof(T)*m_numElements;
		}
		
		template <typename T>
		void DeviceMemPointer_CU<T>::clearDevicePointer()
		{
			cudaFree(m_deviceDataPointer);
			m_deviceDataPointer = NULL;
		}


		/*!
		* Copies all ranges from other device copies (in same or different device memories) and from the main-copy that resides in host memory.
		* A copy plan is passed as argument that specifies what needs to be copied from what source. 
		* TODO: The method may optimize data transfers by overlapping possible communications.
		* \param updateStruct an array of structs containing information about different HTD/DTD/DTH copies that need to be carried out
		* \param sizeUpdStr marks the length of the \p updateStruct array.
		* \param streamID the CUDA Stream ID to possibly overlap HtD transfers with Kernel executions (define USE_MULTI_STREAM and USE_PINNED_MEMORY) 
		*/
		template <typename T>
		void DeviceMemPointer_CU<T>::copyAllRangesToDevice(UpdateInf<T>* updateStruct, const size_t sizeUpdStr, size_t streamID)
		{
			assert(m_valid == false);
	
	// reallocate if datapointer has been cleared
			if (m_deviceDataPointer == NULL)
			{
				cudaError_t er;
				er  = cudaMalloc((void**)&m_deviceDataPointer, m_numElements*sizeof(T));
				if (er == cudaErrorMemoryAllocation)
				{
					freeUpDeviceMem();
				}
			}

			/*! how can we copy into something that has changed... we can copy from something that has changed but not this way??? */
			assert(m_deviceDataHasChanged == false);

			size_t sizeVec;

			for(int i=0; i<sizeUpdStr; ++i)
			{
				if(updateStruct[i].srcIsHost)
				{ 
					DEBUG_TEXT_LEVEL1(m_nameVerbose + " HOST_TO_DEVICE: Host -> GPU_" << m_deviceID << ", size: " << (updateStruct[i].copySize/sizeof(T)) << " # " << updateStruct[i].srcOffset << " -- " << updateStruct[i].srcOffset + (updateStruct[i].copySize/sizeof(T)) <<"!!!\n")
				}
				else
				{
					DEBUG_TEXT_LEVEL1(m_nameVerbose + " DEVICE_TO_DEVICE: From GPU_" << updateStruct[i].srcDevId << " -> GPU_" << m_deviceID << ", size: " << (updateStruct[i].copySize/sizeof(T)) << " # " << updateStruct[i].srcOffset << " -- " << updateStruct[i].srcOffset + (updateStruct[i].copySize/sizeof(T)) <<"!!!\n")
				}
      
				/*!
				* copy could be either from host memory or from some copy in current device
				* memory or from other device memory (possible only when peer-peer memory access enabled...
       */
				enum cudaMemcpyKind memKind = ((updateStruct[i].srcIsHost)? cudaMemcpyHostToDevice : ((updateStruct[i].srcDevId == m_deviceID)? cudaMemcpyDeviceToDevice: cudaMemcpyDefault));

				sizeVec = updateStruct[i].copySize;
				CHECK_CUDA_ERROR(cudaSetDevice(m_deviceID));

#ifdef USE_PINNED_MEMORY
				if(m_usePitch)
					assert(false); /*! dont support yet. TBA in future */
				else if(updateStruct[i].srcIsHost || updateStruct[i].srcDevId == m_deviceID)
					CHECK_CUDA_ERROR(cudaMemcpyAsync(m_deviceDataPointer + updateStruct[i].dstOffset, updateStruct[i].src + updateStruct[i].srcOffset, sizeVec, memKind, (m_dev->m_streams[streamID])))
						else
							CHECK_CUDA_ERROR(cudaMemcpyPeer(m_deviceDataPointer + updateStruct[i].dstOffset, m_deviceID, updateStruct[i].src + updateStruct[i].srcOffset, updateStruct[i].srcDevId, sizeVec));
#else
				if(m_usePitch) /*! dont support yet. TBA in future */
					assert(false);
				else if(updateStruct[i].srcIsHost || updateStruct[i].srcDevId == m_deviceID)
					CHECK_CUDA_ERROR(cudaMemcpy(m_deviceDataPointer + updateStruct[i].dstOffset, updateStruct[i].src + updateStruct[i].srcOffset, sizeVec, memKind))
						else
							CHECK_CUDA_ERROR(cudaMemcpyPeer(m_deviceDataPointer + updateStruct[i].dstOffset, m_deviceID, updateStruct[i].src + updateStruct[i].srcOffset, updateStruct[i].srcDevId, sizeVec));
#endif
			}
#ifdef VERBOSE
			T *h_ptr;
			h_ptr = (T*)malloc(m_numElements*sizeof(T));
			CHECK_CUDA_ERROR(cudaMemcpy(h_ptr, m_deviceDataPointer, m_numElements*sizeof(T), cudaMemcpyDeviceToHost));
			std::cerr << "%% printing '" << m_nameVerbose << "' contents at GPU_" << m_deviceID << "\n";
			for(int i=0; i<m_numElements; ++i)
			{
				std::cerr << h_ptr[i] << " ";
			}
			std::cerr << "\n-----------------------------\n";
			free(h_ptr);
#endif

			m_valid = true;
		}


		/*!
		*  Copies data from host memory to device memory.
		*
		*  \param numElements Number of elements to copy, default value 0 = all elements.
		*/
		template <typename T>
		void DeviceMemPointer_CU<T>::copyHostToDevice(size_t numElements) const
		{
			if(m_hostDataPointer != NULL)
			{
				DEBUG_TEXT_LEVEL1(m_nameVerbose + " HOST_TO_DEVICE: Host -> GPU_" << m_deviceID << ", size: " << ((numElements<1)? m_numElements: numElements)<<" !!!\n")

					if(m_valid == true)
				{
					SKEPU_ERROR("Data copy is already valid.. copying data from host to device failed\n");
				}

				size_t sizeVec;

				// used for pitch allocation.
				int _rows, _cols;

				if(numElements < 1)
				{
					numElements = m_numElements;
				}

				if(m_usePitch)
				{
					if( (numElements%m_cols)!=0 || (numElements/m_cols)<1 ) // using pitch option, memory copy must be proper, respecting rows and cols
					{
						SKEPU_ERROR("Cannot copy data using pitch option when size mismatches with rows and columns. numElements: "<<numElements<<",  rows:"<< m_rows <<", m_cols: "<<m_cols<<"\n");
					}

					_rows = numElements/m_cols;
					_cols = m_cols;
				}

				sizeVec = numElements*sizeof(T);

				CHECK_CUDA_ERROR(cudaSetDevice(m_deviceID));

#ifdef USE_PINNED_MEMORY
				if(m_usePitch)
					CHECK_CUDA_ERROR(cudaMemcpy2DAsync(m_deviceDataPointer,m_pitch*sizeof(T),m_hostDataPointer,_cols*sizeof(T), _cols*sizeof(T), _rows, cudaMemcpyHostToDevice, (m_dev->m_streams[0])))
						else
							CHECK_CUDA_ERROR(cudaMemcpyAsync(m_deviceDataPointer, m_hostDataPointer, sizeVec, cudaMemcpyHostToDevice, (m_dev->m_streams[0])));
#else
				if(m_usePitch)
					CHECK_CUDA_ERROR(cudaMemcpy2D(m_deviceDataPointer,m_pitch*sizeof(T),m_hostDataPointer,_cols*sizeof(T), _cols*sizeof(T), _rows, cudaMemcpyHostToDevice))
						else
							CHECK_CUDA_ERROR(cudaMemcpy(m_deviceDataPointer, m_hostDataPointer, sizeVec, cudaMemcpyHostToDevice));
#endif

				/*! set that the copy is valid now */
				m_valid = true;

				m_deviceDataHasChanged = false;
			}
		}

		/*!
		*  Copies data from device memory to host memory. Only copies if data on device has been marked as changed.
		*
		*  \param numElements Number of elements to copy, default value 0 = all elements.
		*/
		template <typename T>
		void DeviceMemPointer_CU<T>::copyDeviceToHost(size_t numElements) const
		{
			if(m_valid == false)
			{
				SKEPU_ERROR("Data copy is not valid.. copying data from device to host failed: " << ((numElements<1)? m_numElements: numElements) << "\n");
			}

			if(m_deviceDataHasChanged && m_hostDataPointer != NULL)
			{
				DEBUG_TEXT_LEVEL1(m_nameVerbose + " DEVICE_TO_HOST: GPU_" << m_deviceID << " -> Host, size: " << ((numElements<1)? m_numElements: numElements)<<" !!!\n")
				
				size_t sizeVec;
				
				// used for pitch allocation.
				size_t _rows, _cols;
				
				if (numElements < 1)
					numElements = m_numElements;
				
				if (m_usePitch)
				{
					if( (numElements%m_cols)!=0 || (numElements/m_cols)<1 ) // using pitch option, memory copy must be proper, respecting rows and cols
					{
						SKEPU_ERROR("Cannot copy data using pitch option when size mismatches with rows and columns. numElements: "<<numElements<<",  rows:"<< m_rows <<", m_cols: "<<m_cols<<"\n");
					}

					_rows = numElements/m_cols;
					_cols = m_cols;
				}

				sizeVec = numElements*sizeof(T);

				CHECK_CUDA_ERROR(cudaSetDevice(m_deviceID));

#ifdef USE_PINNED_MEMORY
				if(m_usePitch)
					{ CHECK_CUDA_ERROR(cudaMemcpy2DAsync(m_hostDataPointer,_cols*sizeof(T),m_deviceDataPointer,m_pitch*sizeof(T), _cols*sizeof(T), _rows, cudaMemcpyDeviceToHost, (m_dev->m_streams[0]))); }
				else
					{ CHECK_CUDA_ERROR(cudaMemcpyAsync(m_hostDataPointer, m_deviceDataPointer, sizeVec, cudaMemcpyDeviceToHost, (m_dev->m_streams[0]))); }
#else
				if(m_usePitch)
					{ CHECK_CUDA_ERROR(cudaMemcpy2D(m_hostDataPointer,_cols*sizeof(T),m_deviceDataPointer,m_pitch*sizeof(T), _cols*sizeof(T), _rows, cudaMemcpyDeviceToHost)); }
				else
					{ CHECK_CUDA_ERROR(cudaMemcpy(m_hostDataPointer, m_deviceDataPointer, sizeVec, cudaMemcpyDeviceToHost)); }
#endif
      
				m_deviceDataHasChanged = false;
			}
		}

		/*!
		*  \return Pointer to device memory.
		*/
		template <typename T>
		T* DeviceMemPointer_CU<T>::getDeviceDataPointer() const
		{
			return m_deviceDataPointer;
		}

		/*!
		*  \return The device ID of the CUDA device that has the allocation.
		*/
		template <typename T>
		unsigned int DeviceMemPointer_CU<T>::getDeviceID() const
		{
			return m_deviceID;
		}

		/*!
		*  Marks the device data as changed.
		*/
		template <typename T>
		void DeviceMemPointer_CU<T>::changeDeviceData(bool condition)
		{
			if (!condition)
				return;
			
			DEBUG_TEXT_LEVEL2(m_nameVerbose + " DEVICE_DATA_CHANGED: GPU_" << m_deviceID << ", size: " <<  m_numElements <<" !!!\n")
				if(m_valid == false) // this is for data that is directly written on gpu....
			{
				DEBUG_TEXT_LEVEL2(m_nameVerbose + " DEVICE_DATA_MARKED_VALID: GPU_" << m_deviceID << ", size: " <<  m_numElements <<" !!!\n")
					m_valid = true;
			}
			m_deviceDataHasChanged = true;
		}

		/*!
		*  Marks the device data as changed.
		*/
		template <typename T>
		bool DeviceMemPointer_CU<T>::deviceDataHasChanged() const
		{
			return m_deviceDataHasChanged;
		}


		/*!
		*  Marks the copy as invalid. Any further read operation would require first copying data to this copy
		* Also sets modified flag to false...
		*/
		template <typename T>
		void DeviceMemPointer_CU<T>::markCopyInvalid()
		{
			if(m_valid)
			{
				DEBUG_TEXT_LEVEL2(m_nameVerbose + " DEVICE_DATA_MARKED_INVALID: GPU_" << m_deviceID << ", size: " <<  m_numElements <<" !!!\n")
					m_valid = false;
				m_deviceDataHasChanged = false;
			
				DeviceAllocations_CU<int>::getInstance()->removeAllocation(m_deviceDataPointer,this,m_deviceID);
			}
		}

		/*!
		*  Returns whether the copy is valid or not
		*/
		template <typename T>
		bool DeviceMemPointer_CU<T>::isCopyValid() const
		{
			return m_valid;
		}

		// used to free up memory on device after encountering cuda out of memory error
		template <typename T>
		bool DeviceMemPointer_CU<T>::freeUpDeviceMem()
		{	
			DeviceAllocations_CU<int>* dev_alloc = DeviceAllocations_CU<int>::getInstance();
			bool b = dev_alloc->freeAllocation(m_numElements*sizeof(T), m_deviceID);
			if (b)
			{
				cudaMalloc((void**)&m_deviceDataPointer, m_numElements*sizeof(T));
			}
			else
				printf("device out of memory, didnt find a container to free \n");
			return b;	
		}
		
		
		/* Type trait helper */ 
		template<typename T>
		struct to_device_pointer_cu
		{
			using type = typename T::device_pointer_type_cu;
		};
		
		template<typename...Ts>
		struct to_device_pointer_cu<std::tuple<Ts...>>
		{
			using type = std::tuple<typename Ts::device_pointer_type_cu...>;
		};
		
		
		/* Type trait helper */ 
		template<typename T>
		struct to_proxy_cu
		{
			using type = std::pair<typename T::device_pointer_type_cu, typename T::proxy_type>;
		};
		
		template<typename...Ts>
		struct to_proxy_cu<std::tuple<Ts...>>
		{
			using type = std::tuple<std::pair<typename Ts::device_pointer_type_cu, typename Ts::proxy_type>...>;
		};

	}
}

#endif

#endif


