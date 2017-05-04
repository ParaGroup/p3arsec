#undef _GLIBCXX_USE_INT128
#undef _GLIBCXX_ATOMIC_BUILTINS
/*
 * This file is based on source code by NVIDIA.
 * Copyright 1993-2011 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


// StarPU settings...
#define BW_MEASURE_SIZE_IN_BYTES (32*1024*1024*sizeof(char))
#define NITER	128




// defines, project
#define MEMCOPY_ITERATIONS  128 //10
// #define DEFAULT_SIZE        ( 32 * ( 1 << 20 ) )    //32 M
// #define DEFAULT_INCREMENT   (1 << 22)               //4 M
#define CACHE_CLEAR_SIZE    (1 << 24)               //16 M




#ifdef SKEPU_CUDA

namespace skepu2
{
namespace backend
{


//enums, project
enum testMode   { QUICK_MODE, RANGE_MODE, SHMOO_MODE };
enum memcpyKind { DEVICE_TO_HOST, HOST_TO_DEVICE, DEVICE_TO_DEVICE };
// enum printMode  { USER_READABLE, CSV };
enum memoryMode { PINNED, PAGEABLE };



const std::string BW_FILE_PATH("bandwidthMeasures.dat");

#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL( call) {                                    \
   cudaError err = call;                                                  \
   if( SKEPU_UNLIKELY(err) ) {                                              \
       fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString( err) );            \
       exit(EXIT_FAILURE);                                                \
   } }
#endif

///////////////////////////////////////////////////////////////////////////////
//  test the bandwidth of a device to host memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
inline double testDeviceToHostTransfer(unsigned int memSize, memoryMode memMode, bool wc)
{
   float elapsedTimeInMs = 0.0f;
   unsigned char *h_idata = NULL;
   unsigned char *h_odata = NULL;
   cudaEvent_t start, stop;

   CUDA_SAFE_CALL( cudaEventCreate( &start ) );
   CUDA_SAFE_CALL( cudaEventCreate( &stop ) );

   //allocate host memory
   if( PINNED == memMode )
   {
      //pinned memory mode - use special function to get OS-pinned memory
      CUDA_SAFE_CALL( cudaHostAlloc( (void**)&h_idata, memSize, (wc) ? cudaHostAllocWriteCombined : 0 ) );
      CUDA_SAFE_CALL( cudaHostAlloc( (void**)&h_odata, memSize, (wc) ? cudaHostAllocWriteCombined : 0 ) );
   }
   else
   {
      //pageable memory mode - use malloc
      h_idata = (unsigned char *)malloc( memSize );
      h_odata = (unsigned char *)malloc( memSize );

      if( h_idata == 0 || h_odata == 0 )
      {
         fprintf(stderr, "Not enough memory avaialable on host to run test!\n" );
         exit(-1);
      }
   }
   //initialize the memory
   for(unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
   {
      h_idata[i] = (unsigned char) (i & 0xff);
   }

   // allocate device memory
   unsigned char* d_idata;
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata, memSize));

   //initialize the device memory
   CUDA_SAFE_CALL( cudaMemcpy( d_idata, h_idata, memSize, cudaMemcpyHostToDevice) );

   //copy data from GPU to Host
   CUDA_SAFE_CALL( cudaEventRecord( start, 0 ) );
   if( PINNED == memMode )
   {
      for( unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++ )
      {
         CUDA_SAFE_CALL( cudaMemcpyAsync( h_odata, d_idata, memSize, cudaMemcpyDeviceToHost, 0) );
         CUDA_SAFE_CALL( cudaDeviceSynchronize() );
      }
   }
   else
   {
      for( unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++ )
      {
         CUDA_SAFE_CALL( cudaMemcpy( h_odata, d_idata, memSize, cudaMemcpyDeviceToHost) );
         CUDA_SAFE_CALL( cudaDeviceSynchronize() );
      }
   }
   CUDA_SAFE_CALL( cudaEventRecord( stop, 0 ) );

   // make sure GPU has finished copying
     CUDA_SAFE_CALL( cudaDeviceSynchronize() );

   //get the the total elapsed time in ms
   CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTimeInMs, start, stop ) );

   double elapsedTimeInMicroSec = ((double)elapsedTimeInMs*1000)/MEMCOPY_ITERATIONS; // /memSize;

   //calculate bandwidth in MB/s
//     bandwidthInMBs = (1e3f * memSize * (double)MEMCOPY_ITERATIONS) / (elapsedTimeInMs * (double)(1 << 20));

   //clean up memory
   CUDA_SAFE_CALL( cudaEventDestroy(stop) );
   CUDA_SAFE_CALL( cudaEventDestroy(start) );

   if( PINNED == memMode )
   {
      CUDA_SAFE_CALL( cudaFreeHost(h_idata) );
      CUDA_SAFE_CALL( cudaFreeHost(h_odata) );
   }
   else
   {
      free(h_idata);
      free(h_odata);
   }

   CUDA_SAFE_CALL(cudaFree(d_idata));

   return elapsedTimeInMicroSec;
}

///////////////////////////////////////////////////////////////////////////////
//! test the bandwidth of a host to device memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
inline double testHostToDeviceTransfer(unsigned int memSize, memoryMode memMode, bool wc)
{
   float elapsedTimeInMs = 0.0f;
   cudaEvent_t start, stop;
   CUDA_SAFE_CALL( cudaEventCreate( &start ) );
   CUDA_SAFE_CALL( cudaEventCreate( &stop ) );

   //allocate host memory
   unsigned char *h_odata = NULL;
   if( PINNED == memMode )
   {
      //pinned memory mode - use special function to get OS-pinned memory
      CUDA_SAFE_CALL( cudaHostAlloc( (void**)&h_odata, memSize, (wc) ? cudaHostAllocWriteCombined : 0 ) );
   }
   else
   {
      //pageable memory mode - use malloc
      h_odata = (unsigned char *)malloc( memSize );

      if( h_odata == 0 )
      {
         fprintf(stderr, "Not enough memory avaialable on host to run test!\n" );
         exit(-1);
      }
   }

   unsigned char *h_cacheClear1 = (unsigned char *)malloc( CACHE_CLEAR_SIZE );
   unsigned char *h_cacheClear2 = (unsigned char *)malloc( CACHE_CLEAR_SIZE );

   if( h_cacheClear1 == 0 || h_cacheClear1 == 0 )
   {
      fprintf(stderr, "Not enough memory avaialable on host to run test!\n" );
      exit(-1);
   }

   //initialize the memory
   for(unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
   {
      h_odata[i] = (unsigned char) (i & 0xff);
   }
   for(unsigned int i = 0; i < CACHE_CLEAR_SIZE / sizeof(unsigned char); i++)
   {
      h_cacheClear1[i] = (unsigned char) (i & 0xff);
      h_cacheClear2[i] = (unsigned char) (0xff - (i & 0xff));
   }

   //allocate device memory
   unsigned char* d_idata;
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata, memSize));

   CUDA_SAFE_CALL( cudaEventRecord( start, 0 ) );

   //copy host memory to device memory
   if( PINNED == memMode )
   {
      for(unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
      {
         CUDA_SAFE_CALL( cudaMemcpyAsync( d_idata, h_odata, memSize, cudaMemcpyHostToDevice, 0) );
         CUDA_SAFE_CALL( cudaDeviceSynchronize() );
      }
   }
   else
   {
      for(unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
      {
         CUDA_SAFE_CALL( cudaMemcpy( d_idata, h_odata, memSize, cudaMemcpyHostToDevice) );
         CUDA_SAFE_CALL( cudaDeviceSynchronize() );
      }
   }

   CUDA_SAFE_CALL( cudaEventRecord( stop, 0 ) );
     CUDA_SAFE_CALL( cudaDeviceSynchronize() );

   //total elapsed time in ms
   CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTimeInMs, start, stop ) );

   double elapsedTimeInMicroSec = ((double)elapsedTimeInMs*1000)/MEMCOPY_ITERATIONS; // /memSize;

   //calculate bandwidth in MB/s
//     bandwidthInMBs = (1e3f * memSize * (double)MEMCOPY_ITERATIONS) / (elapsedTimeInMs * (double)(1 << 20));

   //clean up memory
   CUDA_SAFE_CALL( cudaEventDestroy(stop) );
   CUDA_SAFE_CALL( cudaEventDestroy(start) );

   if( PINNED == memMode )
   {
      CUDA_SAFE_CALL( cudaFreeHost(h_odata) );
   }
   else
   {
      free(h_odata);
   }

   free(h_cacheClear1);
   free(h_cacheClear2);

   CUDA_SAFE_CALL(cudaFree(d_idata));

   return elapsedTimeInMicroSec;
}

///////////////////////////////////////////////////////////////////////////////
//! test the bandwidth of a device to device memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
inline double testDeviceToDeviceTransfer(unsigned int memSize)
{
   float elapsedTimeInMs = 0.0f;
   cudaEvent_t start, stop;

   CUDA_SAFE_CALL( cudaEventCreate( &start ) );
   CUDA_SAFE_CALL( cudaEventCreate( &stop ) );

   //allocate host memory
   unsigned char *h_idata = (unsigned char *)malloc( memSize );
   if( h_idata == 0 )
   {
      fprintf(stderr, "Not enough memory avaialable on host to run test!\n" );
      exit(-1);
   }

   //initialize the host memory
   for(unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
   {
      h_idata[i] = (unsigned char) (i & 0xff);
   }

   //allocate device memory
   unsigned char *d_idata;
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_idata, memSize));

   unsigned char *d_odata;
   CUDA_SAFE_CALL( cudaMalloc( (void**) &d_odata, memSize));

   //initialize memory
   CUDA_SAFE_CALL( cudaMemcpy( d_idata, h_idata, memSize,
                               cudaMemcpyHostToDevice) );

   //run the memcopy
   CUDA_SAFE_CALL( cudaEventRecord( start, 0 ) );
   for( unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++ )
   {
      CUDA_SAFE_CALL( cudaMemcpy( d_odata, d_idata, memSize, cudaMemcpyDeviceToDevice) );
      CUDA_SAFE_CALL( cudaDeviceSynchronize() );
   }
   CUDA_SAFE_CALL( cudaEventRecord( stop, 0 ) );

   //Since device to device memory copies are non-blocking,
   //cudaDeviceSynchronize() is required in order to get
   //proper timing.
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );

   //get the the total elapsed time in ms
   CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTimeInMs, start, stop ) );

   double elapsedTimeInMicroSec = ((double)elapsedTimeInMs*1000)/MEMCOPY_ITERATIONS; // /memSize;

   //calculate bandwidth in MB/s
//     bandwidthInMBs = 2.0f * (1e3f * memSize * (double)MEMCOPY_ITERATIONS) / (elapsedTimeInMs * (double)(1 << 20));

   //clean up memory
   free(h_idata);
   CUDA_SAFE_CALL(cudaEventDestroy(stop));
   CUDA_SAFE_CALL(cudaEventDestroy(start));
   CUDA_SAFE_CALL(cudaFree(d_idata));
   CUDA_SAFE_CALL(cudaFree(d_odata));

   return elapsedTimeInMicroSec;
}

/////////////////////////////////////////////////////////
//print results in an easily read format
////////////////////////////////////////////////////////
inline void printResultsReadable(unsigned int *memSizes, double* bandwidths, unsigned int count, memcpyKind kind, memoryMode memMode, int deviceId, bool wc)
{
   // log config information
   if (kind == DEVICE_TO_DEVICE)
   {
      printf(" Device to Device Bandwidth, Device %i, \n", deviceId);
   }
   else
   {
      if (kind == DEVICE_TO_HOST)
      {
         printf(" Device to Host Bandwidth, Device %i, ", deviceId);
      }
      else if (kind == HOST_TO_DEVICE)
      {
         printf(" Host to Device Bandwidth, Device %i, ", deviceId);
      }

      if(memMode == PAGEABLE)
      {
         printf("Paged memory\n");
      }
      else if (memMode == PINNED)
      {
         printf("Pinned memory");
         if (wc)
         {
            printf(", Write-Combined Memory Enabled");
         }
         printf("\n");
      }
   }

   printf("   Transfer Size (Bytes)\tBandwidth(MB/s)\n");
   unsigned int i;
   for(i = 0; i < (count - 1); i++)
   {
      printf("   %u\t\t\t%s%.1f\n", memSizes[i], (memSizes[i] < 10000)? "\t" : "", bandwidths[i]);
   }
   printf("   %u\t\t\t%s%.1f\n\n", memSizes[i], (memSizes[i] < 10000)? "\t" : "", bandwidths[i]);
}








///////////////////////////////////////////////////////////////////////
//  Run a bandwidth test
//////////////////////////////////////////////////////////////////////
inline DevTimingStruct measurebandwidth(memoryMode memMode = PAGEABLE, int deviceId = 0, bool wc = false)
{
   DevTimingStruct devTiming;

   int size = BW_MEASURE_SIZE_IN_BYTES;


   // Use the device asked by the user
   cudaSetDevice(deviceId);

   //run each of the copies
   devTiming.timing_dth = testDeviceToHostTransfer( size, memMode, wc);
   devTiming.timing_htd = testHostToDeviceTransfer( size, memMode, wc);
   devTiming.timing_dtd = testDeviceToDeviceTransfer( size );

   devTiming.latency_dth = testDeviceToHostTransfer( 1, memMode, wc);
   devTiming.latency_htd = testHostToDeviceTransfer( 1, memMode, wc);
   devTiming.latency_dtd = testDeviceToDeviceTransfer( 1 );

   // now divide the measure bandwidth time to be time for 1024 bytes...
   int sizeOf1024 = BW_MEASURE_SIZE_IN_BYTES/1024;
   devTiming.timing_dth /= sizeOf1024;
   devTiming.timing_htd /= sizeOf1024;
   devTiming.timing_dtd /= sizeOf1024;

   // Ensure that we reset the CUDA Device in question
   cudaSetDevice(deviceId);
   cudaDeviceReset();

   return devTiming;
}

/*!
 * This function is called to measure or load bandwidth for a GPU
 * \param gpuId ID of the GPU for which to measure bandwidth
 * \param pinnedMemory flag specifying whether to measure via pinned memory or not
 * \return A structure containing bandwidth measurement information.
 */
inline DevTimingStruct measureOrLoadCUDABandwidth(int gpuId, bool pinnedMemory = false)
{
   DevTimingStruct devBW;
   bool readFromFile = false;

   // first try to load from a file if it exists...
   std::ifstream infile(BW_FILE_PATH.c_str());
   if(infile.good())
   {
      std::string strLine;
      int id;
      std::istringstream iss;
      do
      {
         getline(infile, strLine);
         iss.str(strLine);
         iss >> id;
         if(id == gpuId)
         {
            iss >> devBW.timing_htd >> devBW.timing_dth >> devBW.timing_dtd >> devBW.latency_htd >> devBW.latency_dth >> devBW.latency_dtd;
            readFromFile = true;
            infile.close();
            break;
         }
      }
      while(infile.good());
   }

   if(!readFromFile)
   {
      bool wc = false;

      memoryMode memMode = (pinnedMemory ? PINNED : PAGEABLE);

      devBW = measurebandwidth(memMode, gpuId, wc);

      // now try to save it in the file for next usage... first try to append may be measurements does not exist for this gpuid else write it...
      std::ofstream outfile(BW_FILE_PATH.c_str(), std::ios_base::app | std::ios_base::ate);
      if(!outfile.good())
         outfile.open(BW_FILE_PATH.c_str());

      if(!(outfile.good()))
      {
         SKEPU_ERROR("Could not open file for writing/appending. filename = " << BW_FILE_PATH);
      }

      outfile << gpuId << " " << devBW.timing_htd << " " << devBW.timing_dth << " " << devBW.timing_dtd << " " << devBW.latency_htd << " " << devBW.latency_dth << " " << devBW.latency_dtd << "\n";
      outfile.close();
   }

   // IN file when saving as well as when measuring we measure bandwidth for 1024 bytes as number becomes too small if consider for 1 byte (precision issue when saving/loading)
   // however when giving it to the application we give it per byte form...
   devBW.timing_dth /= 1024;
   devBW.timing_htd /= 1024;
   devBW.timing_dtd /= 1024;

   return devBW;
}



}
}

#endif


