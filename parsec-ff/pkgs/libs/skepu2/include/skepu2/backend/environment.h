/*! \file environment.h
 *  \brief Contains a class declaration for Environment class.
 */

#ifndef DEVICE_H
#define DEVICE_H

#include <vector>
#include <sstream>

#include <skepu2/backend/globals.h>

#ifdef SKEPU_PRECOMPILED
#include <skepu2/backend/device_cl.h>
#include <skepu2/backend/device_mem_pointer_cl.h>
#include <skepu2/backend/device_cu.h>
#include <skepu2/backend/device_mem_pointer_cu.h>
#include <skepu2/backend/skepu_opencl_helpers.h>
#endif // SKEPU_PRECOMPILED

namespace skepu2
{
namespace backend
{

   
/*!
 *  \ingroup helpers
 */

enum ImplType
{
   IMPL_CPU,
   IMPL_OMP,
   IMPL_CUDA,
   IMPL_OPENCL
};



enum skepu_container_type
{
   VECTOR=0,
   MATRIX=1
};


/*!
 *  \ingroup helpers
 */

// Forward declaration
template <typename T>
class EnvironmentDestroyer;

/*!
 *  \class Environment
 *
 *  \brief A class representing a execution environment.
 *
 *  The Environment is used by the skeleton objects to define a execution environment for them to use. It mainly keeps track of
 *  which devices are available and gives access to them. It is implemented as a singleton so that only one environment is actually used
 *  and the skeletons stores a pointer to this instance which is created by the first defined skeleton.
 */
template <typename T>
class Environment
{

public:
   static Environment* getInstance();


#ifdef SKEPU_PRECOMPILED

#ifdef SKEPU_OPENCL

   void finishAll_CL();
   std::vector<Device_CL*> m_devices_CL;
   
#endif // SKEPU_OPENCL
   
   unsigned int m_numDevices;

#ifdef SKEPU_CUDA

   unsigned int bestCUDADevID;
   
   int m_peerAccessEnabled; /*! 0 means not enabled, 1 means enabled between all gpu combinations, -1 means enabled between some of the GPUs */
   std::vector<std::pair<int, int> > m_peerCopyGpuIDsVector;
   
   bool supportsCUDAOverlap();
   
   void finishAll_CU(int lowID=-1, int highID=-1);

   std::vector<Device_CU*> m_devices_CU;

#endif // SKEPU_CUDA


   DevTimingStruct bwDataStruct;


   bool getGroupMapping(int groupId, Backend::Type &type);

   void addGroupMapping(int groupId, Backend::Type type);

   void clearGroupMapping();

protected:

   /*!
   * This attribute allows multiple skeleton implementations to be scheduled on the same backend */
   std::vector<std::pair<int, Backend::Type> > m_groupMapping;
   std::pair<int, Backend::Type> m_cacheGroupResult;

   Environment();

   virtual ~Environment();

   friend class EnvironmentDestroyer<T>; // To safely clean resources



public:

#ifdef SKEPU_OPENCL
   void createOpenCLProgramForMatrixTranspose();
   std::vector<std::pair<cl_kernel, Device_CL*> > m_transposeKernels_CL;
#endif // SKEPU_OPENCL


#endif // SKEPU_PRECOMPILED

   void finishAll();

private:

   static EnvironmentDestroyer<T> _destroyer;


   void init();

#ifdef SKEPU_OPENCL
   void init_CL();
#endif // SKEPU_OPENCL

#ifdef SKEPU_CUDA
   void init_CU();
#endif // SKEPU_CUDA



private:
   static Environment* instance;

};

template <typename T>
EnvironmentDestroyer<T> Environment<T>::_destroyer;


/*!
 *  \ingroup hekpers
 */

/*!
 *  \class EnvironmentDestroyer
 *
 *  \brief A class that is used to properly deallocate singelton object of Environment class.
 *
 *  This class is used to simplify the destruction of Environment static object which acts like a
 *  singelton object.
 */
template <typename T>
class EnvironmentDestroyer
{
public:
   EnvironmentDestroyer(Environment<T>* = 0);
   ~EnvironmentDestroyer();

   void SetEnvironment(Environment<T>* s);
private:
   Environment<T>* _singleton;
   
};



template <typename T>
EnvironmentDestroyer<T>::EnvironmentDestroyer(Environment<T>* s)
{
   _singleton = s;
}

template <typename T>
EnvironmentDestroyer<T>::~EnvironmentDestroyer ()
{
   delete _singleton;
}

template <typename T>
void EnvironmentDestroyer<T>::SetEnvironment (Environment<T>* s)
{
   _singleton = s;
}

}
}


#ifdef SKEPU_PRECOMPILED
#include "impl/environment.inl"
#endif // SKEPU_PRECOMPILED

#endif // DEVICE_H
