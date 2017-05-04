#include <algorithm>

#include "code_gen.h"

using namespace clang;

// ------------------------------
// Kernel templates
// ------------------------------

/*!
 *
 *  OpenCL MapOverlap kernel for vector. It uses one device thread per element so maximum number of device threads
 *  limits the maximum number of elements this kernel can handle. Also the amount of shared memory and
 *  the maximum blocksize limits the maximum overlap that is possible to use, typically limits the
 *  overlap to < 256.
 */
static const std::string MapOverlapKernel_CU = R"~~~(
__global__ void SKEPU_KERNEL_NAME_MapOverlapKernel_CU(
	SKEPU_MAPOVERLAP_INPUT_TYPE* input, SKEPU_KERNEL_PARAMS SKEPU_MAPOVERLAP_RESULT_TYPE* output,
	SKEPU_MAPOVERLAP_INPUT_TYPE* wrap, size_t n,
	size_t out_offset, size_t out_numelements,
	int poly, SKEPU_MAPOVERLAP_INPUT_TYPE pad, size_t overlap
)
{
   extern __shared__ char _sdata[];
   SKEPU_MAPOVERLAP_INPUT_TYPE* sdata = reinterpret_cast<SKEPU_MAPOVERLAP_INPUT_TYPE*>(_sdata);

   size_t tid = threadIdx.x;
   size_t i = blockIdx.x * blockDim.x + threadIdx.x;
   size_t gridSize = blockDim.x*gridDim.x;

   while(i<(n+overlap-1))
   {
      //Copy data to shared memory
      if(poly == 0) // constant policy
      {
         sdata[overlap+tid] = (i < n) ? input[i] : pad;

         if(tid < overlap)
         {
            sdata[tid] = (i<overlap) ? pad : input[i-overlap];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = (i+overlap < n) ? input[i+overlap] : pad;
         }
      }
      else if(poly == 1)
      {
         if(i < n)
         {
            sdata[overlap+tid] = input[i];
         }
         else
         {
            sdata[overlap+tid] = wrap[overlap+(i-n)];
         }

         if(tid < overlap)
         {
            sdata[tid] = (i<overlap) ? wrap[tid] : input[i-overlap];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = (i+overlap < n) ? input[i+overlap] : wrap[overlap+(i+overlap-n)];
         }
      }
      else if(poly == 2) // DUPLICATE
      {
         sdata[overlap+tid] = (i < n) ? input[i] : input[n-1];

         if(tid < overlap)
         {
            sdata[tid] = (i<overlap) ? input[0] : input[i-overlap];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = (i+overlap < n) ? input[i+overlap] : input[n-1];
         }
      }

      __syncthreads();

      //Compute and store data
      if( (i >= out_offset) && (i < out_offset+out_numelements) )
      	 output[i-out_offset] = SKEPU_FUNCTION_NAME_MAPOVERLAP(overlap, 1, &sdata[tid + overlap] SKEPU_MAPOVERLAP_ARGS);

      i += gridSize;

      __syncthreads();
   }
}
)~~~";


/*!
 *
 *  OpenCL MapOverlap kernel for applying row-wise overlap on matrix operands. It
 *  uses one device thread per element so maximum number of device threads
 *  limits the maximum number of elements this kernel can handle. Also the amount of shared memory and
 *  the maximum blocksize limits the maximum overlap that is possible to use, typically limits the
 *  overlap to < 256.
 */
static const std::string MapOverlapKernel_CU_Matrix_Row = R"~~~(
__global__ void SKEPU_KERNEL_NAME_MapOverlapKernel_CU_Matrix_Row(
	SKEPU_MAPOVERLAP_INPUT_TYPE* input, SKEPU_KERNEL_PARAMS SKEPU_MAPOVERLAP_RESULT_TYPE* output,
	SKEPU_MAPOVERLAP_INPUT_TYPE* wrap, size_t n, size_t out_offset, size_t out_numelements,
	int poly, SKEPU_MAPOVERLAP_INPUT_TYPE pad, size_t overlap, size_t blocksPerRow, size_t rowWidth
)
{
   extern __shared__ char _sdata[];
   SKEPU_MAPOVERLAP_INPUT_TYPE* sdata = reinterpret_cast<SKEPU_MAPOVERLAP_INPUT_TYPE*>(_sdata);

   size_t tid = threadIdx.x;
   size_t i = blockIdx.x * blockDim.x + tid;

   size_t wrapIndex= 2 * overlap * (int)(blockIdx.x/blocksPerRow);
   size_t tmp= (blockIdx.x % blocksPerRow);
   size_t tmp2= (blockIdx.x / blocksPerRow);


   //Copy data to shared memory
   if(poly == 0)
   {
      sdata[overlap+tid] = (i < n) ? input[i] : pad;

      if(tid < overlap)
      {
         sdata[tid] = (tmp==0) ? pad : input[i-overlap];
      }

      if(tid >= (blockDim.x-overlap))
      {
         sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (i+overlap < n) && tmp!=(blocksPerRow-1)) ? input[i+overlap] : pad;
      }
   }
   else if(poly == 1)
   {
      if(i < n)
      {
         sdata[overlap+tid] = input[i];
      }
      else if(i-n < overlap)
      {
         sdata[overlap+tid] = wrap[(overlap+(i-n))+ wrapIndex];
      }
      else
      {
         sdata[overlap+tid] = pad;
      }

      if(tid < overlap)
      {
         sdata[tid] = (tmp==0) ? wrap[tid+wrapIndex] : input[i-overlap];
      }

      if(tid >= (blockDim.x-overlap))
      {
         sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && i+overlap < n && tmp!=(blocksPerRow-1)) ? input[i+overlap] : wrap[overlap+wrapIndex+(tid+overlap-blockDim.x)];
      }
   }
   else if(poly == 2)
   {
      sdata[overlap+tid] = (i < n) ? input[i] : input[n-1];

      if(tid < overlap)
      {
         sdata[tid] = (tmp==0) ? input[tmp2*rowWidth] : input[i-overlap];
      }

      if(tid >= (blockDim.x-overlap))
      {
         sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (i+overlap < n) && (tmp!=(blocksPerRow-1))) ? input[i+overlap] : input[(tmp2+1)*rowWidth-1];
      }
   }

   __syncthreads();

   //Compute and store data
   if( (i >= out_offset) && (i < out_offset+out_numelements) )
   	output[i-out_offset] = SKEPU_FUNCTION_NAME_MAPOVERLAP(overlap, 1, &sdata[tid + overlap] SKEPU_MAPOVERLAP_ARGS);
}
)~~~";



/*!
 *
 *  OpenCL MapOverlap kernel for applying column-wise overlap on matrix operands. It
 *  uses one device thread per element so maximum number of device threads
 *  limits the maximum number of elements this kernel can handle. Also the amount of shared memory and
 *  the maximum blocksize limits the maximum overlap that is possible to use, typically limits the
 *  overlap to < 256.
 */
static const std::string MapOverlapKernel_CU_Matrix_Col = R"~~~(
__global__ void SKEPU_KERNEL_NAME_MapOverlapKernel_CU_Matrix_Col(
	SKEPU_MAPOVERLAP_INPUT_TYPE* input, SKEPU_KERNEL_PARAMS SKEPU_MAPOVERLAP_RESULT_TYPE* output,
	SKEPU_MAPOVERLAP_INPUT_TYPE* wrap, size_t n, size_t out_offset, size_t out_numelements,
	int poly, SKEPU_MAPOVERLAP_INPUT_TYPE pad, size_t overlap, size_t blocksPerCol, size_t rowWidth, size_t colWidth
)
{
   extern __shared__ char _sdata[];
   SKEPU_MAPOVERLAP_INPUT_TYPE* sdata = reinterpret_cast<SKEPU_MAPOVERLAP_INPUT_TYPE*>(_sdata);

   size_t tid = threadIdx.x;
   size_t i = blockIdx.x * blockDim.x + tid;

   size_t wrapIndex= 2 * overlap * (int)(blockIdx.x/blocksPerCol);
   size_t tmp= (blockIdx.x % blocksPerCol);
   size_t tmp2= (blockIdx.x / blocksPerCol);

   size_t arrInd = (threadIdx.x + tmp*blockDim.x)*rowWidth + ((blockIdx.x)/blocksPerCol);

   //Copy data to shared memory
   if(poly == 0)
   {
      sdata[overlap+tid] = (i < n) ? input[arrInd] : pad;

      if(tid < overlap)
      {
         sdata[tid] = (tmp==0) ? pad : input[(arrInd-(overlap*rowWidth))];
      }

      if(tid >= (blockDim.x-overlap))
      {
         sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+(overlap*rowWidth))] : pad;
      }
   }
   else if(poly == 1)
   {
      if(i < n)
      {
         sdata[overlap+tid] = input[arrInd];
      }
      else if(i-n < overlap)
      {
         sdata[overlap+tid] = wrap[(overlap+(i-n))+ wrapIndex];
      }
      else
      {
         sdata[overlap+tid] = pad;
      }

      if(tid < overlap)
      {
         sdata[tid] = (tmp==0) ? wrap[tid+wrapIndex] : input[(arrInd-(overlap*rowWidth))];
      }

      if(tid >= (blockDim.x-overlap))
      {
         sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+(overlap*rowWidth))] : wrap[overlap+wrapIndex+(tid+overlap-blockDim.x)];
      }
   }
   else if(poly == 2)
   {
      sdata[overlap+tid] = (i < n) ? input[arrInd] : input[n-1];

      if(tid < overlap)
      {
         sdata[tid] = (tmp==0) ? input[tmp2] : input[(arrInd-(overlap*rowWidth))];
      }

      if(tid >= (blockDim.x-overlap))
      {
         sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+(overlap*rowWidth))] : input[tmp2+(colWidth-1)*rowWidth];
      }
   }

   __syncthreads();

   //Compute and store data
   if( (arrInd >= out_offset) && (arrInd < out_offset+out_numelements) )
   {
      output[arrInd-out_offset] = SKEPU_FUNCTION_NAME_MAPOVERLAP(overlap, 1, &sdata[tid + overlap] SKEPU_MAPOVERLAP_ARGS);
   }
}
)~~~";



/*!
 *
 *  OpenCL MapOverlap kernel for applying column-wise overlap on matrix operands when using multiple GPUs. It
 *  uses one device thread per element so maximum number of device threads
 *  limits the maximum number of elements this kernel can handle. Also the amount of shared memory and
 *  the maximum blocksize limits the maximum overlap that is possible to use, typically limits the
 *  overlap to < 256.
 */
static const std::string MapOverlapKernel_CU_Matrix_ColMulti = R"~~~(
__global__ void SKEPU_KERNEL_NAME_MapOverlapKernel_CU_Matrix_ColMulti(
	SKEPU_MAPOVERLAP_INPUT_TYPE* input, SKEPU_KERNEL_PARAMS SKEPU_MAPOVERLAP_RESULT_TYPE* output,
	SKEPU_MAPOVERLAP_INPUT_TYPE* wrap, size_t n, size_t in_offset, size_t out_numelements,
	int poly, int deviceType, SKEPU_MAPOVERLAP_INPUT_TYPE pad, size_t overlap, size_t blocksPerCol, size_t rowWidth, size_t colWidth
)
{
   extern __shared__ char _sdata[];
   SKEPU_MAPOVERLAP_INPUT_TYPE* sdata = reinterpret_cast<SKEPU_MAPOVERLAP_INPUT_TYPE*>(_sdata);

   size_t tid = threadIdx.x;
   size_t i = blockIdx.x * blockDim.x + tid;

   size_t tmp= (blockIdx.x % blocksPerCol);
   size_t tmp2= (blockIdx.x / blocksPerCol);

   size_t arrInd = (threadIdx.x + tmp*blockDim.x)*rowWidth + tmp2; //((blockIdx.x)/blocksPerCol);

   if(poly == 0) //IF overlap policy is CONSTANT
   {
      sdata[overlap+tid] = (i < n) ? input[arrInd+in_offset] : pad; // in_offset

      if(deviceType == -1) // first device, i.e. in_offset=0
      {
         if(tid < overlap)
         {
            sdata[tid] = (tmp==0) ? pad : input[(arrInd-(overlap*rowWidth))];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if(deviceType == 0) // middle device
      {
         if(tid < overlap)
         {
            sdata[tid] = input[arrInd];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if(deviceType == 1) // last device
      {
         if(tid < overlap)
         {
            sdata[tid] = input[arrInd];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+in_offset+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+in_offset+(overlap*rowWidth))] : pad;
         }
      }
   }
   else if(poly == 1) //IF overlap policy is CYCLIC
   {
      sdata[overlap+tid] = (i < n) ? input[arrInd+in_offset] : ((i-n < overlap) ? wrap[(i-n)+ (overlap * tmp2)] : pad);

      if(deviceType == -1) // first device, i.e. in_offset=0
      {
         if(tid < overlap)
         {
            sdata[tid] = (tmp==0) ? wrap[tid+(overlap * tmp2)] : input[(arrInd-(overlap*rowWidth))];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if(deviceType == 0) // middle device
      {
         if(tid < overlap)
         {
            sdata[tid] = input[arrInd];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if(deviceType == 1) // last device
      {
         if(tid < overlap)
         {
            sdata[tid] = input[arrInd];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+in_offset+(overlap*rowWidth))] : wrap[(overlap * tmp2)+(tid+overlap-blockDim.x)];
         }
      }
   }
   else if(poly == 2) //IF overlap policy is DUPLICATE
   {
      sdata[overlap+tid] = (i < n) ? input[arrInd+in_offset] : input[n+in_offset-1];

      if(deviceType == -1) // first device, i.e. in_offset=0
      {
         if(tid < overlap)
         {
            sdata[tid] = (tmp==0) ? input[tmp2] : input[(arrInd-(overlap*rowWidth))];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if(deviceType == 0) // middle device
      {
         if(tid < overlap)
         {
            sdata[tid] = input[arrInd];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if(deviceType == 1) // last device
      {
         if(tid < overlap)
         {
            sdata[tid] = input[arrInd];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+in_offset+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+in_offset+(overlap*rowWidth))] : input[tmp2+in_offset+(colWidth-1)*rowWidth];
         }
      }
   }

   __syncthreads();

   //Compute and store data
   if( arrInd < out_numelements )
   {
      output[arrInd] = SKEPU_FUNCTION_NAME_MAPOVERLAP(overlap, 1, &sdata[tid + overlap] SKEPU_MAPOVERLAP_ARGS);
   }
}
)~~~";


std::string createMapOverlap1DKernelProgram_CU(UserFunction &mapOverlapFunc, std::string dir)
{
	const std::string kernelName = ResultName + "_OverlapKernel_" + mapOverlapFunc.uniqueName;
	
	std::stringstream SSMapOverlapFuncArgs, SSKernelParamList;
	
	for (UserFunction::RandomAccessParam& param : mapOverlapFunc.anyContainerParams)
	{
		SSKernelParamList << param.fullTypeName << " " << param.name << ", ";
		SSMapOverlapFuncArgs << ", " << param.name;
	}
	
	for (UserFunction::Param& param : mapOverlapFunc.anyScalarParams)
	{
		SSKernelParamList << param.resolvedTypeName << " " << param.name << ", ";
		SSMapOverlapFuncArgs << ", " << param.name;
	}
	
	const Type *type = mapOverlapFunc.elwiseParams[2].astDeclNode->getOriginalType().getTypePtr();
	const PointerType *pointer = dyn_cast<PointerType>(type);
	QualType pointee = pointer->getPointeeType().getUnqualifiedType();
	std::string elemType = pointee.getAsString();
	
	std::string kernelSource = MapOverlapKernel_CU + MapOverlapKernel_CU_Matrix_Row + MapOverlapKernel_CU_Matrix_Col + MapOverlapKernel_CU_Matrix_ColMulti;
	replaceTextInString(kernelSource, PH_MapOverlapInputType, elemType);
	replaceTextInString(kernelSource, PH_MapOverlapResultType, mapOverlapFunc.resolvedReturnTypeName);
	replaceTextInString(kernelSource, PH_KernelName, kernelName);
	replaceTextInString(kernelSource, PH_MapOverlapFuncName, mapOverlapFunc.funcNameCUDA());
	replaceTextInString(kernelSource, PH_KernelParams, SSKernelParamList.str());
	replaceTextInString(kernelSource, PH_MapOverlapArgs, SSMapOverlapFuncArgs.str());
	
	std::ofstream FSOutFile {dir + "/" + kernelName + ".cu"};
	FSOutFile << kernelSource;
	return kernelName;
}




/*!
* The mapoverlap OpenCL kernel to apply a user function on neighbourhood of each element in the matrix.
*/
static const std::string MatrixConvol2D_CU = R"~~~(
__global__ void SKEPU_KERNEL_NAME_conv_cuda_2D_kernel(
	SKEPU_MAPOVERLAP_INPUT_TYPE* input, SKEPU_KERNEL_PARAMS SKEPU_MAPOVERLAP_RESULT_TYPE* output,
	const size_t out_rows, const size_t out_cols,
	size_t overlap_y, size_t overlap_x,
	size_t in_pitch, size_t out_pitch,
	const size_t sharedRows, const size_t sharedCols
)
{
	extern __shared__ char _sdata[];
	SKEPU_MAPOVERLAP_INPUT_TYPE* sdata = reinterpret_cast<SKEPU_MAPOVERLAP_INPUT_TYPE*>(_sdata); // will also contain extra (overlap data)
	
	size_t xx = blockIdx.x * blockDim.x;
	size_t yy = blockIdx.y * blockDim.y;
	
	size_t x = xx + threadIdx.x;
	size_t y = yy + threadIdx.y;
	
	if (x < out_cols + overlap_x * 2 && y < out_rows + overlap_y * 2)
	{
		sdata[threadIdx.y * sharedCols + threadIdx.x] = input[y * in_pitch + x];
		
		// To load data in shared memory including neighbouring elements...
		for (size_t shared_y = threadIdx.y; shared_y < sharedRows; shared_y += blockDim.y)
		{
			for (size_t shared_x = threadIdx.x; shared_x < sharedCols; shared_x += blockDim.x)
			{
				sdata[shared_y * sharedCols + shared_x] = input[(yy + shared_y) * in_pitch + xx + shared_x];
			}
		}
	}
	
	__syncthreads();
	
	if (x < out_cols && y < out_rows)
		output[y*out_pitch+x] = SKEPU_FUNCTION_NAME_MAPOVERLAP(overlap_x, overlap_y,
			sharedCols, &sdata[(threadIdx.y + overlap_y) * sharedCols + (threadIdx.x + overlap_x)] SKEPU_MAPOVERLAP_ARGS);
}
)~~~";


std::string createMapOverlap2DKernelProgram_CU(UserFunction &mapOverlapFunc, std::string dir)
{
	const std::string kernelName = ResultName + "_Overlap2DKernel_" + mapOverlapFunc.uniqueName;
	
	std::stringstream SSMapOverlapFuncArgs, SSKernelParamList;
	
	for (UserFunction::RandomAccessParam& param : mapOverlapFunc.anyContainerParams)
	{
		SSKernelParamList << param.fullTypeName << " " << param.name << ", ";
		SSMapOverlapFuncArgs << ", " << param.name;
	}
	
	for (UserFunction::Param& param : mapOverlapFunc.anyScalarParams)
	{
		SSKernelParamList << param.resolvedTypeName << " " << param.name << ", ";
		SSMapOverlapFuncArgs << ", " << param.name;
	}
	
	const Type *type = mapOverlapFunc.elwiseParams[3].astDeclNode->getOriginalType().getTypePtr();
	const PointerType *pointer = dyn_cast<PointerType>(type);
	QualType pointee = pointer->getPointeeType().getUnqualifiedType();
	std::string elemType = pointee.getAsString();
	
	std::string kernelSource = MatrixConvol2D_CU;
	replaceTextInString(kernelSource, PH_MapOverlapInputType, elemType);
	replaceTextInString(kernelSource, PH_MapOverlapResultType, mapOverlapFunc.resolvedReturnTypeName);
	replaceTextInString(kernelSource, PH_KernelName, kernelName);
	replaceTextInString(kernelSource, PH_MapOverlapFuncName, mapOverlapFunc.funcNameCUDA());
	replaceTextInString(kernelSource, PH_KernelParams, SSKernelParamList.str());
	replaceTextInString(kernelSource, PH_MapOverlapArgs, SSMapOverlapFuncArgs.str());
	
	std::ofstream FSOutFile {dir + "/" + kernelName + ".cu"};
	FSOutFile << kernelSource;
	return kernelName;
}
