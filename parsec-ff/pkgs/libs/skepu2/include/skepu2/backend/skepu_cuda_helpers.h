/*! \file skepu_cuda_helpers.h
 *  \brief Contains the definitions of some helper functions related to \em CUDA backend.
 */

#ifndef SKEPU_CUDA_HELPER_H
#define SKEPU_CUDA_HELPER_H

#ifdef SKEPU_CUDA

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <cufft.h>
#include <curand.h>




// We define these calls here, so the user doesn't need to include __FILE__ and __LINE__
// The advantage is the developers gets to use the inline function so they can debug
#define cutilSafeCallNoSync(err)     __cudaSafeCallNoSync(err, __FILE__, __LINE__)
#define cutilSafeCall(err)           __cudaSafeCall      (err, __FILE__, __LINE__)
#define cutilSafeThreadSync()        __cudaSafeThreadSync(__FILE__, __LINE__)
#define cufftSafeCall(err)           __cufftSafeCall     (err, __FILE__, __LINE__)
#define curandSafeCall(err)          __curandSafeCall    (err, __FILE__, __LINE__)
#define cutilCheckError(err)         __cutilCheckError   (err, __FILE__, __LINE__)
#define cutilCheckMsg(msg)           __cutilGetLastError (msg, __FILE__, __LINE__)
#define cutilCheckMsgAndSync(msg)    __cutilGetLastErrorAndSync (msg, __FILE__, __LINE__)
#define cutilSafeMalloc(mallocCall)  __cutilSafeMalloc   ((mallocCall), __FILE__, __LINE__)
#define cutilCondition(val)          __cutilCondition    (val, __FILE__, __LINE__)
#define cutilExit(argc, argv)        __cutilExit         (argc, argv)


#define CHECK_CUDA_ERROR(stmt) { cudaError_t err = stmt; __checkCudaError (err, __FILE__, __LINE__); }


// #define CHECK_CUDA_ERROR(stmt) {cudaError_t err = stmt; if(err != cudaSuccess){std::cerr<<"CUDA Error at " << __FILE__ << ":" << __LINE__ << " => " << cudaGetErrorString(err) << "\n"; }}
////////////////////////////////////////////////////////////////////////////
//! CUT bool type
////////////////////////////////////////////////////////////////////////////

#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif


inline cudaError cutilDeviceSynchronize()
{
#if CUDART_VERSION >= 4000
   return cudaDeviceSynchronize();
#else
   return cudaThreadSynchronize();
#endif
}


////////////////////////////////////////////////////////////////////////////
//! CUT bool type
////////////////////////////////////////////////////////////////////////////
enum CUTBoolean
{
   CUTFalse = 0,
   CUTTrue = 1
};

#ifdef _WIN32
#define CUTIL_API __stdcall
#else
#define CUTIL_API
#endif





// Give a little more for Windows : the console window often disapears before we can read the message
#ifdef _WIN32
# if 1//ndef UNICODE
#  ifdef _DEBUG // Do this only in debug mode...
inline void VSPrintf(FILE *file, LPCSTR fmt, ...)
{
   size_t fmt2_sz	= 2048;
   char *fmt2		= (char*)malloc(fmt2_sz);
   va_list  vlist;
   va_start(vlist, fmt);
   while((_vsnprintf(fmt2, fmt2_sz, fmt, vlist)) < 0) // means there wasn't anough room
   {
      fmt2_sz *= 2;
      if(fmt2) free(fmt2);
      fmt2 = (char*)malloc(fmt2_sz);
   }
   OutputDebugStringA(fmt2);
   fprintf(file, fmt2);
   free(fmt2);
}
#	define FPRINTF(a) VSPrintf a
#  else //debug
#	define FPRINTF(a) fprintf a
// For other than Win32
#  endif //debug
# else //unicode
// Unicode case... let's give-up for now and keep basic printf
#	define FPRINTF(a) fprintf a
# endif //unicode
#else //win32
#	define FPRINTF(a) fprintf a
#endif //win32


// NOTE: "%s(%i) : " allows Visual Studio to directly jump to the file at the right line
// when the user double clicks on the error line in the Output pane. Like any compile error.

inline void __checkCudaError( cudaError_t err, const char *file, const int line )
{
   if( cudaSuccess != err)
   {
      FPRINTF((stderr, "CUDA ERROR at %s: %i. Error is %d: %s.\n",file, line, (int)err, cudaGetErrorString(err)));
   }
}



inline void __cudaSafeCallNoSync( cudaError_t err, const char *file, const int line )
{
   if( cudaSuccess != err)
   {
      FPRINTF((stderr, "%s(%i) : cudaSafeCallNoSync() Runtime API error %d : %s.\n",
               file, line, (int)err, cudaGetErrorString( err ) ));
      exit(-1);
   }
}

inline void __cudaSafeCall( cudaError_t err, const char *file, const int line )
{
   if( cudaSuccess != err)
   {
      FPRINTF((stderr, "%s(%i) : cudaSafeCall() Runtime API error %d: %s.\n",
               file, line, (int)err, cudaGetErrorString( err ) ));
      exit(-1);
   }
}



inline void __cudaSafeThreadSync( const char *file, const int line )
{
   cudaError_t err = cutilDeviceSynchronize();
   if ( cudaSuccess != err)
   {
      FPRINTF((stderr, "%s(%i) : cudaDeviceSynchronize() Runtime API error %d: %s.\n",
               file, line, (int)err, cudaGetErrorString( err ) ));
      exit(-1);
   }
}

inline void __cufftSafeCall( cufftResult err, const char *file, const int line )
{
   if( CUFFT_SUCCESS != err)
   {
      FPRINTF((stderr, "%s(%i) : cufftSafeCall() CUFFT error %d: ",
               file, line, (int)err));
      switch (err)
      {
      case CUFFT_INVALID_PLAN:
         FPRINTF((stderr, "CUFFT_INVALID_PLAN\n"));
      case CUFFT_ALLOC_FAILED:
         FPRINTF((stderr, "CUFFT_ALLOC_FAILED\n"));
      case CUFFT_INVALID_TYPE:
         FPRINTF((stderr, "CUFFT_INVALID_TYPE\n"));
      case CUFFT_INVALID_VALUE:
         FPRINTF((stderr, "CUFFT_INVALID_VALUE\n"));
      case CUFFT_INTERNAL_ERROR:
         FPRINTF((stderr, "CUFFT_INTERNAL_ERROR\n"));
      case CUFFT_EXEC_FAILED:
         FPRINTF((stderr, "CUFFT_EXEC_FAILED\n"));
      case CUFFT_SETUP_FAILED:
         FPRINTF((stderr, "CUFFT_SETUP_FAILED\n"));
      case CUFFT_INVALID_SIZE:
         FPRINTF((stderr, "CUFFT_INVALID_SIZE\n"));
      case CUFFT_UNALIGNED_DATA:
         FPRINTF((stderr, "CUFFT_UNALIGNED_DATA\n"));
      default:
         FPRINTF((stderr, "CUFFT Unknown error code\n"));
      }
      exit(-1);
   }
}

inline void __curandSafeCall( curandStatus_t err, const char *file, const int line )
{
   if( CURAND_STATUS_SUCCESS != err)
   {
      FPRINTF((stderr, "%s(%i) : curandSafeCall() CURAND error %d: ",
               file, line, (int)err));
      switch (err)
      {
      case CURAND_STATUS_VERSION_MISMATCH:
         FPRINTF((stderr, "CURAND_STATUS_VERSION_MISMATCH"));
      case CURAND_STATUS_NOT_INITIALIZED:
         FPRINTF((stderr, "CURAND_STATUS_NOT_INITIALIZED"));
      case CURAND_STATUS_ALLOCATION_FAILED:
         FPRINTF((stderr, "CURAND_STATUS_ALLOCATION_FAILED"));
      case CURAND_STATUS_TYPE_ERROR:
         FPRINTF((stderr, "CURAND_STATUS_TYPE_ERROR"));
      case CURAND_STATUS_OUT_OF_RANGE:
         FPRINTF((stderr, "CURAND_STATUS_OUT_OF_RANGE"));
      case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
         FPRINTF((stderr, "CURAND_STATUS_LENGTH_NOT_MULTIPLE"));
//             case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
//                                                     FPRINTF((stderr, "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED"));
      case CURAND_STATUS_LAUNCH_FAILURE:
         FPRINTF((stderr, "CURAND_STATUS_LAUNCH_FAILURE"));
      case CURAND_STATUS_PREEXISTING_FAILURE:
         FPRINTF((stderr, "CURAND_STATUS_PREEXISTING_FAILURE"));
      case CURAND_STATUS_INITIALIZATION_FAILED:
         FPRINTF((stderr, "CURAND_STATUS_INITIALIZATION_FAILED"));
      case CURAND_STATUS_ARCH_MISMATCH:
         FPRINTF((stderr, "CURAND_STATUS_ARCH_MISMATCH"));
      case CURAND_STATUS_INTERNAL_ERROR:
         FPRINTF((stderr, "CURAND_STATUS_INTERNAL_ERROR"));
      default:
         FPRINTF((stderr, "CURAND Unknown error code\n"));
      }
      exit(-1);
   }
}


inline void __cutilCheckError( CUTBoolean err, const char *file, const int line )
{
   if( CUTTrue != err)
   {
      FPRINTF((stderr, "%s(%i) : CUTIL CUDA error.\n",
               file, line));
      exit(-1);
   }
}

inline void __cutilGetLastError( const char *errorMessage, const char *file, const int line )
{
   cudaError_t err = cudaGetLastError();
   if( cudaSuccess != err)
   {
      FPRINTF((stderr, "%s(%i) : cutilCheckMsg() CUTIL CUDA error : %s : (%d) %s.\n",
               file, line, errorMessage, (int)err, cudaGetErrorString( err ) ));
      exit(-1);
   }
}

inline void __cutilGetLastErrorAndSync( const char *errorMessage, const char *file, const int line )
{
   cudaError_t err = cudaGetLastError();
   if( cudaSuccess != err)
   {
      FPRINTF((stderr, "%s(%i) : cutilCheckMsg() CUTIL CUDA error : %s : (%d) %s.\n",
               file, line, errorMessage, (int)err, cudaGetErrorString( err ) ));
      exit(-1);
   }

   err = cutilDeviceSynchronize();
   if( cudaSuccess != err)
   {
      FPRINTF((stderr, "%s(%i) : cutilCheckMsg cudaDeviceSynchronize error: %s : (%d) %s.\n",
               file, line, errorMessage, (int)err, cudaGetErrorString( err ) ));
      exit(-1);
   }
}

inline void __cutilSafeMalloc( void *pointer, const char *file, const int line )
{
   if( !(pointer))
   {
      FPRINTF((stderr, "%s(%i) : cutilSafeMalloc host malloc failure\n",
               file, line));
      exit(-1);
   }
}


// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores_local(int major, int minor)
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
      {
         return nGpuArchCoresPerSM[index].Cores;
      }
      index++;
   }
   SKEPU_WARNING("MapSMtoCores undefined SMversion " << major << "." << minor << "\n");
   return -1;
}
// end of GPU Architecture definitions

// This function returns the best GPU (with maximum GFLOPS)
inline int cutGetMaxGflopsDeviceId()
{
   int current_device   = 0, sm_per_multiproc = 0;
   int max_compute_perf = 0, max_perf_device  = 0;
   int device_count     = 0, best_SM_arch     = 0;
   cudaDeviceProp deviceProp;

   cudaGetDeviceCount( &device_count );
   // Find the best major SM Architecture GPU device
   while ( current_device < device_count )
   {
      cudaGetDeviceProperties( &deviceProp, current_device );
      if (deviceProp.major > 0 && deviceProp.major < 9999)
      {
         best_SM_arch = MAX(best_SM_arch, deviceProp.major);
      }
      current_device++;
   }

   // Find the best CUDA capable GPU device
   current_device = 0;
   while( current_device < device_count )
   {
      cudaGetDeviceProperties( &deviceProp, current_device );
      if (deviceProp.major == 9999 && deviceProp.minor == 9999)
      {
         sm_per_multiproc = 1;
      }
      else
      {
         sm_per_multiproc = _ConvertSMVer2Cores_local(deviceProp.major, deviceProp.minor);
      }

      int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
      if( compute_perf  > max_compute_perf )
      {
         // If we find GPU with SM major > 2, search only these
         if ( best_SM_arch > 2 )
         {
            // If our device==dest_SM_arch, choose this, or else pass
            if (deviceProp.major == best_SM_arch)
            {
               max_compute_perf  = compute_perf;
               max_perf_device   = current_device;
            }
         }
         else
         {
            max_compute_perf  = compute_perf;
            max_perf_device   = current_device;
         }
      }
      ++current_device;
   }
   return max_perf_device;
}



// General initialization call to pick the best CUDA Device
inline int cutilChooseCudaDevice()
{
   cudaDeviceProp deviceProp;
   int devID = 0;

   // Otherwise pick the device with highest Gflops/s
   devID = cutGetMaxGflopsDeviceId();
   cutilSafeCallNoSync( cudaSetDevice( devID ) );
   cutilSafeCallNoSync( cudaGetDeviceProperties(&deviceProp, devID) );
   DEBUG_TEXT_LEVEL1("Best CUDA device [" << devID << "]: " << deviceProp.name << "\n");

   return devID;
}



#ifdef USE_PINNED_MEMORY
template <typename T>
void copyDeviceToHost(T *hostPtr, T *devPtr, int numElements, cudaStream_t &stream)
#else
template <typename T>
void copyDeviceToHost(T *hostPtr, T *devPtr, int numElements)
#endif
{
   if(devPtr != NULL && hostPtr != NULL)
   {
      DEBUG_TEXT_LEVEL2("** DEVICE_TO_HOST CUDA: "<< numElements <<"!!!\n")

      size_t sizeVec;

      sizeVec = numElements*sizeof(T);

#ifdef USE_PINNED_MEMORY
      cutilSafeCallNoSync( cudaMemcpyAsync(hostPtr, devPtr, sizeVec, cudaMemcpyDeviceToHost, stream) );
#else
      cutilSafeCallNoSync( cudaMemcpy(hostPtr, devPtr, sizeVec, cudaMemcpyDeviceToHost) );
#endif
   }
}



#ifdef USE_PINNED_MEMORY
template <typename T>
void copyHostToDevice(T *hostPtr, T *devPtr, int numElements, cudaStream_t &stream)
#else
template <typename T>
void copyHostToDevice(T *hostPtr, T *devPtr, int numElements)
#endif
{
   if(hostPtr != NULL && devPtr != NULL)
   {
      DEBUG_TEXT_LEVEL2("** HOST_TO_DEVICE CUDA: "<< numElements <<"!!!\n")

      size_t sizeVec;

      sizeVec = numElements*sizeof(T);

#ifdef USE_PINNED_MEMORY
      cutilSafeCallNoSync( cudaMemcpyAsync(devPtr, hostPtr, sizeVec, cudaMemcpyHostToDevice, stream) );
#else
      cutilSafeCallNoSync( cudaMemcpy(devPtr, hostPtr, sizeVec, cudaMemcpyHostToDevice) );
#endif
   }
}


template <typename T>
inline void allocateCudaMemory(T **devicePointer, unsigned int size)
{
   DEBUG_TEXT_LEVEL2("** ALLOC CUDA: "<< size <<"!!!\n")

   size_t sizeVec = size*sizeof(T);

   cutilSafeCallNoSync( cudaMalloc((void**)devicePointer, sizeVec) );
}


template <typename T>
inline void freeCudaMemory(T *d_pointer)
{
   DEBUG_TEXT_LEVEL2("** DE-ALLOC CUDA: !!!\n")

   if(d_pointer!=NULL)
      cutilSafeCallNoSync(cudaFree(d_pointer));
}





#endif

#endif
