/*! \file device_cu.h
 *  \brief Contains a class declaration for the object that represents a CUDA device.
 */

#ifndef DEVICE_CU_H
#define DEVICE_CU_H

#ifdef SKEPU_CUDA

#include <iostream>
#include <cuda.h>

#include "globals.h"



namespace skepu2
{
namespace backend
{

/*!
 *  \ingroup helpers
 */

/*!
 *  \class Device_CU
 *
 *  \brief A class representing a CUDA device.
 *
 *  This class represents one CUDA device which can be used to execute the skeletons on if CUDA
 *  is used as backend. Stores various properties about the device and provides functions that return them.
 */
class Device_CU
{

public:
   cudaStream_t m_streams[MAX_POSSIBLE_CUDA_STREAMS_PER_GPU];

private:
   unsigned int m_deviceID;
   cudaDeviceProp m_deviceProp;
   size_t m_maxThreads;
   size_t m_maxBlocks;

   unsigned int m_noConcurrKernelsSupported;

   unsigned int m_noCoresSupported;

   /*!
    *  Run once during construction to get all the device properties.
    *
    *  \param device Integer specifying the device to fetch properties for.
    */
   void initDeviceProps(unsigned int device)
   {
      cudaError_t err;
      err = cudaGetDeviceProperties(&m_deviceProp, device);
      if (err != cudaSuccess)
      {
         SKEPU_ERROR("getDeviceProps failed!\n");
      }

      if (m_deviceProp.major == 9999 && m_deviceProp.minor == 9999)
      {
         m_noConcurrKernelsSupported = 1;
         m_noCoresSupported = 1;
      }
      else
      {
         m_noConcurrKernelsSupported = getMaxConcurKernelsSupported(m_deviceProp.major, m_deviceProp.minor);
         if(m_noConcurrKernelsSupported > MAX_POSSIBLE_CUDA_STREAMS_PER_GPU)
         {
            SKEPU_WARNING("Potential problem as stream size specified is larger tham what is maximum possible specified in MAX_POSSIBLE_CUDA_STREAMS_PER_GPU.\n");
            m_noConcurrKernelsSupported = MAX_POSSIBLE_CUDA_STREAMS_PER_GPU; // reset it to max as we have allocated stream array of size MAX_POSSIBLE_CUDA_STREAMS_PER_GPU
         }

         m_noCoresSupported = ConvertSMVer2Cores_local(m_deviceProp.major, m_deviceProp.minor);

      }
   }


   /*!
    * Returns number of cores on the current GPU
    *
    * \param major the major version number of current GPU compute architecture.
   * \param minor the minor version number of current GPU compute architecture.
   * \return the number of cores on the current GPU.
    */
   int ConvertSMVer2Cores_local(int major, int minor)
   {
      // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
      typedef struct
      {
         int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
         int Cores;
      } sSMtoCores;

      sSMtoCores nGpuArchCoresPerSM[] =
      {
         { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
         { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
         { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
         { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
         { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
         { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
         { 0x30, 192}, // Fermi Generation (SM 3.0) GK10x class
         { 0x35, 192}, // ???
         {   -1, -1 }
      };

      int index = 0;
      while (nGpuArchCoresPerSM[index].SM != -1)
      {
         if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) )
            return nGpuArchCoresPerSM[index].Cores;

         index++;
      }
      SKEPU_WARNING("MapSMtoCores undefined SMversion " << major << "," << minor << "\n");
      return -1;
   }



   /*!
    * \brief returns maximum number of concurrent kernels supported.
    *
    * \param major the major version number of current GPU compute architecture.
    * \param minor the minor version number of current GPU compute architecture.
    * \return the maximum concurrent kernels suppoted on current GPU.
    */
   int getMaxConcurKernelsSupported(int major, int minor)
   {
      // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
      typedef struct
      {
         int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
         int totConcurrKernels;
      } sSMtoCores;

      sSMtoCores nGpuArchCoresPerSM[] =
      {
         { 0x10,  1 }, // Tesla Generation (SM 1.0) G80 class
         { 0x11,  1 }, // Tesla Generation (SM 1.1) G8x class
         { 0x12,  1 }, // Tesla Generation (SM 1.2) G9x class
         { 0x13,  1 }, // Tesla Generation (SM 1.3) GT200 class
         { 0x20,  4 }, // Fermi Generation (SM 2.0) GF100 class
         { 0x21, 16 }, // Fermi Generation (SM 2.1) GF10x class
         { 0x30, 16}, // Kepler Generation (SM 3.0) GK10x class
				 { 0x32, 4}, // special kepler? (SM 3.2)
				 { 0x35, 32}, // Kepler Generation (SM 3.5) GK11x class
				 { 0x37, 32}, // (SM 3.7)  
				 { 0x50, 32}, // Maxwell Generation (SM 5.0) GM10x class 
				 { 0x52, 32}, // Maxwell Generation (SM 5.2) GM20x class
         {   -1, 1 }
      };

      int index = 0;
      while (nGpuArchCoresPerSM[index].SM != -1)
      {
         if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) )
            return nGpuArchCoresPerSM[index].totConcurrKernels;

         index++;
      }
      SKEPU_WARNING("MapSMtoCores undefined SMversion " << major << "," << minor << "\n");
      return 1;
   }


public:

   /*!
    *  The constructor creates a device from an ID and gets all its properties.
    *
    *  \param id Device ID for the device that is to be created.
    */
   Device_CU(unsigned int id)
   {
      m_deviceID = id;

      cudaSetDevice(m_deviceID);

      initDeviceProps(id);

#ifdef USE_PINNED_MEMORY
      for(unsigned int i=0; i<m_noConcurrKernelsSupported; i++)
         cudaStreamCreate(&(m_streams[i]));
#endif

      if(m_deviceProp.major == 1 && m_deviceProp.minor < 2)
      {
         m_maxThreads = 256;
      }
      else
      {
         m_maxThreads = m_deviceProp.maxThreadsPerBlock;
      }

      m_maxBlocks = m_deviceProp.maxGridSize[0];
   }

   /*!
    * \brief The destructor.
    */
   ~Device_CU()
   {
      // Explicitly destroys and cleans up all resources associated with the current device in the current process.
      // Any subsequent API call to this device will reinitialize the device.
      cudaSetDevice(m_deviceID);
      cudaDeviceReset();
   };


   /*!
    * Returns whether the device supports overlap (memory,kernel) operation or not
    */
   bool isOverlapSupported()
   {
      return m_deviceProp.deviceOverlap;
   }

   /*!
    *  \return The maximum block size.
    */
   size_t getMaxBlockSize() const
   {
      return m_deviceProp.maxThreadsPerBlock;
   }

   /*!
    *  \return The major version.
    */
   int getMajorVersion() const
   {
      return m_deviceProp.major;
   }

   /*!
    *  \return The minor version.
    */
   int getMinorVersion() const
   {
      return m_deviceProp.minor;
   }

   /*!
    *  \return The number of streaming processors per multiprocessor for current GPU.
    */
   unsigned int getSmPerMultiProc() const
   {
      return m_noCoresSupported;
   }

   /*!
    *  \return The name of current GPU.
    */
   std::string getDeviceName() const
   {
      return m_deviceProp.name;
   }

   /*!
    * test
    * 
    *  \return The clock rate of current GPU.
    */
   int getClockRate() const
   {
      return m_deviceProp.clockRate;
   }

   /*!
    *  \return The integer specifying whether the overlap is support between pinned memory transfer and kernel launches (value>0 if supported)
    *          and/or between pinned memory HTD and DTH (value=2 if supported).
    */
   int getAsyncEngineCount() const
   {
      return m_deviceProp.asyncEngineCount;
   }

   /*!
    *  \return The boolean specifying whether the concurrent kernels are supported on this GPU or not. If supported maximum kernels count is 16.
    */
   bool IsConcurrentKernels() const
   {
      return m_deviceProp.concurrentKernels;
   }

   /*!
    *  \return The integer specifying the max number of concurrent kernels supported on this GPU (if not supprted then return 1).
    */
   unsigned int getNoConcurrentKernels() const
   {
      return m_noConcurrKernelsSupported;
   }

   /*!
    *  \return The maximum number of compute units available.
    */
   int getNumComputeUnits() const
   {
      return m_deviceProp.multiProcessorCount;
   }

   /*!
    *  \return The global memory size.
    */
   size_t getGlobalMemSize() const
   {
      return m_deviceProp.totalGlobalMem;
   }

   /*!
    *  \return The shared memory size.
    */
   size_t getSharedMemPerBlock() const
   {
      return m_deviceProp.sharedMemPerBlock;
   }

   /*!
    *  \return The maximum number of threads per block or group.
    */
   size_t getMaxThreads() const
   {
#ifdef SKEPU_MAX_GPU_THREADS
      return SKEPU_MAX_GPU_THREADS;
#else
      return m_maxThreads;
#endif
   }

   /*!
    *  \return The maximum number of blocks or groups for a kernel launch.
    */
   size_t getMaxBlocks() const
   {
#ifdef SKEPU_MAX_GPU_BLOCKS
      return SKEPU_MAX_GPU_BLOCKS;
#else
      return m_maxBlocks;
#endif
   }

   /*!
    *  \return Device ID.
    */
   unsigned int getDeviceID() const
   {
      return m_deviceID;
   }
};

}
}

#endif

#endif


