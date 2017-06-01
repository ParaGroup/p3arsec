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
static const std::string MapOverlapKernel_CL = R"~~~(
__kernel void SKEPU_KERNEL_NAME_Vector(
	__global SKEPU_MAPOVERLAP_INPUT_TYPE* input, SKEPU_KERNEL_PARAMS __global SKEPU_MAPOVERLAP_RESULT_TYPE* output,
	__global SKEPU_MAPOVERLAP_INPUT_TYPE* wrap, size_t n, size_t overlap, size_t out_offset,
	size_t out_numelements, int poly, SKEPU_MAPOVERLAP_INPUT_TYPE pad, __local SKEPU_MAPOVERLAP_INPUT_TYPE* sdata
)
{
	size_t tid = get_local_id(0);
	size_t i = get_group_id(0) * get_local_size(0) + get_local_id(0);
	SKEPU_CONTAINER_PROXIES
	
	if (poly == 0)
	{
		sdata[overlap + tid] = (i < n) ? input[i] : pad;
		if (tid < overlap)
			sdata[tid] = (get_group_id(0) == 0) ? pad : input[i - overlap];
		
		if (tid >= get_local_size(0) - overlap)
			sdata[tid + 2 * overlap] = (get_group_id(0) != get_num_groups(0) - 1 && i + overlap < n) ? input[i + overlap] : pad;
	}
	else if (poly == 1)
	{
		if (i < n)
			sdata[overlap + tid] = input[i];
		else if (i - n < overlap)
			sdata[overlap + tid] = wrap[overlap + i - n];
		else
			sdata[overlap + tid] = pad;
		
		if (tid < overlap)
			sdata[tid] = (get_group_id(0) == 0) ? wrap[tid] : input[i - overlap];
		
		if (tid >= get_local_size(0) - overlap)
			sdata[tid + 2 * overlap] = (get_group_id(0) != get_num_groups(0) - 1 && i + overlap < n) ? input[i + overlap] : wrap[overlap + i + overlap - n];
	}
	else if (poly == 2)
	{
		sdata[overlap+tid] = (i < n) ? input[i] : input[n-1];
		if (tid < overlap)
			sdata[tid] = (get_group_id(0) == 0) ? input[0] : input[i-overlap];
		
		if (tid >= get_local_size(0) - overlap)
			sdata[tid + 2 * overlap] = (get_group_id(0) != get_num_groups(0) - 1 && i + overlap < n) ? input[i + overlap] : input[n - 1];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (i >= out_offset && i < out_offset + out_numelements)
		output[i - out_offset] = SKEPU_FUNCTION_NAME_MAPOVERLAP(overlap, 1, &sdata[tid + overlap] SKEPU_MAPOVERLAP_ARGS);
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
static const std::string MapOverlapKernel_CL_Matrix_Row = R"~~~(
__kernel void SKEPU_KERNEL_NAME_MatRowWise(
	__global SKEPU_MAPOVERLAP_INPUT_TYPE* input, SKEPU_KERNEL_PARAMS __global SKEPU_MAPOVERLAP_RESULT_TYPE* output,
	__global SKEPU_MAPOVERLAP_INPUT_TYPE* wrap, size_t n, size_t overlap, size_t out_offset, size_t out_numelements,
	int poly, SKEPU_MAPOVERLAP_INPUT_TYPE pad, size_t blocksPerRow, size_t rowWidth, __local SKEPU_MAPOVERLAP_INPUT_TYPE* sdata
)
{
	size_t tid = get_local_id(0);
	size_t i = get_group_id(0) * get_local_size(0) + get_local_id(0);
	size_t wrapIndex= 2 * overlap * (int)(get_group_id(0) / blocksPerRow);
	size_t tmp  = (get_group_id(0) % blocksPerRow);
	size_t tmp2 = (get_group_id(0) / blocksPerRow);
	SKEPU_CONTAINER_PROXIES
	
	if (poly == 0)
	{
		sdata[overlap+tid] = (i < n) ? input[i] : pad;
		if (tid < overlap)
			sdata[tid] = (tmp==0) ? pad : input[i-overlap];
		
		if (tid >= (get_local_size(0)-overlap))
			sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (i+overlap < n) && tmp!=(blocksPerRow-1)) ? input[i+overlap] : pad;
	}
	else if (poly == 1)
	{
		if (i < n)
			sdata[overlap+tid] = input[i];
		else if (i-n < overlap)
			sdata[overlap+tid] = wrap[(overlap+(i-n))+ wrapIndex];
		else
			sdata[overlap+tid] = pad;
		
		if (tid < overlap)
			sdata[tid] = (tmp==0) ? wrap[tid+wrapIndex] : input[i-overlap];
		
		if (tid >= (get_local_size(0)-overlap))
			sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && i+overlap < n && tmp!=(blocksPerRow-1))
				? input[i+overlap] : wrap[overlap+wrapIndex+(tid+overlap-get_local_size(0))];
	}
	else if (poly == 2)
	{
		sdata[overlap+tid] = (i < n) ? input[i] : input[n-1];
		if(tid < overlap)
			sdata[tid] = (tmp==0) ? input[tmp2*rowWidth] : input[i-overlap];
		
		if(tid >= (get_local_size(0)-overlap))
			sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (i+overlap < n) && (tmp!=(blocksPerRow-1)))
				? input[i+overlap] : input[(tmp2+1)*rowWidth-1];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	if ((i >= out_offset) && (i < out_offset+out_numelements))
		output[i-out_offset] = SKEPU_FUNCTION_NAME_MAPOVERLAP(overlap, 1, &sdata[tid+overlap] SKEPU_MAPOVERLAP_ARGS);
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
static const std::string MapOverlapKernel_CL_Matrix_Col = R"~~~(
__kernel void SKEPU_KERNEL_NAME_MatColWise(
	__global SKEPU_MAPOVERLAP_INPUT_TYPE* input, SKEPU_KERNEL_PARAMS __global SKEPU_MAPOVERLAP_RESULT_TYPE* output,
	__global SKEPU_MAPOVERLAP_INPUT_TYPE* wrap, size_t n, size_t overlap, size_t out_offset, size_t out_numelements,
	int poly, SKEPU_MAPOVERLAP_INPUT_TYPE pad, size_t blocksPerCol, size_t rowWidth, size_t colWidth, __local SKEPU_MAPOVERLAP_INPUT_TYPE* sdata
	)
{
	size_t tid = get_local_id(0);
	size_t i = get_group_id(0) * get_local_size(0) + get_local_id(0);
	size_t wrapIndex= 2 * overlap * (int)(get_group_id(0)/blocksPerCol);
	size_t tmp= (get_group_id(0) % blocksPerCol);
	size_t tmp2= (get_group_id(0) / blocksPerCol);
	size_t arrInd = (tid + tmp*get_local_size(0))*rowWidth + tmp2;
	SKEPU_CONTAINER_PROXIES
	
	if (poly == 0)
	{
		sdata[overlap+tid] = (i < n) ? input[arrInd] : pad;
		if (tid < overlap)
			sdata[tid] = (tmp==0) ? pad : input[(arrInd-(overlap*rowWidth))];
		
		if (tid >= (get_local_size(0)-overlap))
			sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+(overlap*rowWidth))] : pad;
	}
	else if (poly == 1)
	{
		if (i < n)
			sdata[overlap+tid] = input[arrInd];
		else if (i-n < overlap)
			sdata[overlap+tid] = wrap[(overlap+(i-n))+ wrapIndex];
		else
			sdata[overlap+tid] = pad;
		
		if (tid < overlap)
			sdata[tid] = (tmp==0) ? wrap[tid+wrapIndex] : input[(arrInd-(overlap*rowWidth))];
		
		if (tid >= (get_local_size(0)-overlap))
			sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1)))
				? input[(arrInd+(overlap*rowWidth))] : wrap[overlap+wrapIndex+(tid+overlap-get_local_size(0))];
	}
	else if (poly == 2)
	{
		sdata[overlap+tid] = (i < n) ? input[arrInd] : input[n-1];
		if (tid < overlap)
			sdata[tid] = (tmp==0) ? input[tmp2] : input[(arrInd-(overlap*rowWidth))];
		
		if (tid >= (get_local_size(0)-overlap))
			sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1)))
				? input[(arrInd+(overlap*rowWidth))] : input[tmp2+(colWidth-1)*rowWidth];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	if ((arrInd >= out_offset) && (arrInd < out_offset+out_numelements))
		output[arrInd-out_offset] = SKEPU_FUNCTION_NAME_MAPOVERLAP(overlap, 1, &sdata[tid+overlap] SKEPU_MAPOVERLAP_ARGS);
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
static const std::string MapOverlapKernel_CL_Matrix_ColMulti = R"~~~(
__kernel void SKEPU_KERNEL_NAME_MatColWiseMulti(
	__global SKEPU_MAPOVERLAP_INPUT_TYPE* input, SKEPU_KERNEL_PARAMS __global SKEPU_MAPOVERLAP_RESULT_TYPE* output,
	__global SKEPU_MAPOVERLAP_INPUT_TYPE* wrap, size_t n, size_t overlap, size_t in_offset, size_t out_numelements,
	int poly, int deviceType, SKEPU_MAPOVERLAP_INPUT_TYPE pad, size_t blocksPerCol, size_t rowWidth, size_t colWidth,
	__local SKEPU_MAPOVERLAP_INPUT_TYPE* sdata
)
{
	size_t tid = get_local_id(0);
	size_t i   = get_group_id(0) * get_local_size(0) + get_local_id(0);
	size_t wrapIndex = 2 * overlap * (int)(get_group_id(0)/blocksPerCol);
	size_t tmp  = (get_group_id(0) % blocksPerCol);
	size_t tmp2 = (get_group_id(0) / blocksPerCol);
	size_t arrInd = (tid + tmp*get_local_size(0))*rowWidth + tmp2;
	SKEPU_CONTAINER_PROXIES
	
	if (poly == 0)
	{
		sdata[overlap+tid] = (i < n) ? input[arrInd+in_offset] : pad;
		if (deviceType == -1)
		{
			if (tid < overlap)
				sdata[tid] = (tmp==0) ? pad : input[(arrInd-(overlap*rowWidth))];
			 
			if(tid >= (get_local_size(0)-overlap))
				sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
		}
		else if (deviceType == 0) 
		{
			if(tid < overlap)
				sdata[tid] = input[arrInd];
			
			if(tid >= (get_local_size(0)-overlap))
				sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
		}
		else if (deviceType == 1)
		{
			if (tid < overlap)
				sdata[tid] = input[arrInd];
			
			if (tid >= (get_local_size(0)-overlap))
				sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1)))
					? input[(arrInd+in_offset+(overlap*rowWidth))] : pad;
		}
	}
	else if (poly == 1)
	{
		sdata[overlap+tid] = (i < n) ? input[arrInd+in_offset] : ((i-n < overlap) ? wrap[(i-n)+ (overlap * tmp2)] : pad);
		if (deviceType == -1)
		{
			if (tid < overlap)
				sdata[tid] = (tmp==0) ? wrap[tid+(overlap * tmp2)] : input[(arrInd-(overlap*rowWidth))];
			
			if (tid >= (get_local_size(0)-overlap))
				sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
		}
		else if (deviceType == 0)
		{
			if (tid < overlap)
				sdata[tid] = input[arrInd];
			
			if (tid >= (get_local_size(0)-overlap))
				sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
		}
		else if (deviceType == 1)
		{
			if (tid < overlap)
				sdata[tid] = input[arrInd];
			
			if (tid >= (get_local_size(0)-overlap))
				sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1)))
					? input[(arrInd+in_offset+(overlap*rowWidth))] : wrap[(overlap * tmp2)+(tid+overlap-get_local_size(0))];
		}
	}
	else if (poly == 2)
	{
		sdata[overlap+tid] = (i < n) ? input[arrInd + in_offset] : input[n + in_offset - 1];
		if (deviceType == -1)
		{
			if (tid < overlap)
				sdata[tid] = (tmp == 0) ? input[tmp2] : input[arrInd - overlap * rowWidth];
			
			if (tid >= get_local_size(0) - overlap)
				sdata[tid+2*overlap] = input[arrInd + in_offset + overlap * rowWidth];
		}
		else if (deviceType == 0)
		{
			if (tid < overlap)
				sdata[tid] = input[arrInd];
			
			if (tid >= get_local_size(0) - overlap)
				sdata[tid+2*overlap] = input[arrInd + in_offset + overlap * rowWidth];
		}
		else if (deviceType == 1)
		{
			if (tid < overlap)
				sdata[tid] = input[arrInd];
			
			if (tid >= get_local_size(0) - overlap)
				sdata[tid + 2 * overlap] = (get_group_id(0) != get_num_groups(0) - 1 && (arrInd + overlap * rowWidth < n) && (tmp != blocksPerCol - 1))
					? input[arrInd + in_offset + overlap * rowWidth] : input[tmp2 + in_offset + (colWidth - 1) * rowWidth];
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	if (arrInd < out_numelements )
		output[arrInd] = SKEPU_FUNCTION_NAME_MAPOVERLAP(overlap, 1, &sdata[tid+overlap] SKEPU_MAPOVERLAP_ARGS);
}
)~~~";


const std::string Constructor1D = R"~~~(
class SKEPU_KERNEL_CLASS
{
public:
	
	enum
	{
		KERNEL_VECTOR = 0,
		KERNEL_MATRIX_ROW,
		KERNEL_MATRIX_COL,
		KERNEL_MATRIX_COL_MULTI,
		KERNEL_COUNT
	};
	
	static cl_kernel kernels(size_t deviceID, size_t kerneltype, cl_kernel *newkernel = nullptr)
	{
		static cl_kernel arr[8][KERNEL_COUNT]; // Hard-coded maximum
		if (newkernel)
		{
			arr[deviceID][kerneltype] = *newkernel;
			return nullptr;
		}
		else return arr[deviceID][kerneltype];
	}
	
	static void initialize()
	{
		static bool initialized = false;
		if (initialized)
			return;
		
		std::string source = skepu2::backend::cl_helpers::replaceSizeT(R"###(SKEPU_OPENCL_KERNEL)###");
		
		// Builds the code and creates kernel for all devices
		size_t counter = 0;
		for (skepu2::backend::Device_CL *device : skepu2::backend::Environment<int>::getInstance()->m_devices_CL)
		{
			cl_int err;
			cl_program program = skepu2::backend::cl_helpers::buildProgram(device, source);
			cl_kernel kernel_vector = clCreateKernel(program, "SKEPU_KERNEL_NAME_Vector", &err);
			CL_CHECK_ERROR(err, "Error creating MapOverlap 1D vector kernel '" << "SKEPU_KERNEL_NAME" << "'");
			
			cl_kernel kernel_matrix_row = clCreateKernel(program, "SKEPU_KERNEL_NAME_MatRowWise", &err);
			CL_CHECK_ERROR(err, "Error creating MapOverlap 1D matrix row-wise kernel '" << "SKEPU_KERNEL_NAME" << "'");
			
			cl_kernel kernel_matrix_col = clCreateKernel(program, "SKEPU_KERNEL_NAME_MatColWise", &err);
			CL_CHECK_ERROR(err, "Error creating MapOverlap 1D matrix col-wise kernel '" << "SKEPU_KERNEL_NAME" << "'");
			
			cl_kernel kernel_matrix_col_multi = clCreateKernel(program, "SKEPU_KERNEL_NAME_MatColWiseMulti", &err);
			CL_CHECK_ERROR(err, "Error creating MapOverlap 1D matrix col-wise multi kernel '" << "SKEPU_KERNEL_NAME" << "'");
			
			kernels(counter, KERNEL_VECTOR,           &kernel_vector);
			kernels(counter, KERNEL_MATRIX_ROW,       &kernel_matrix_row);
			kernels(counter, KERNEL_MATRIX_COL,       &kernel_matrix_col);
			kernels(counter, KERNEL_MATRIX_COL_MULTI, &kernel_matrix_col_multi);
			counter++;
		}
		
		initialized = true;
	}
	
	static void mapOverlapVector
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		skepu2::backend::DeviceMemPointer_CL<SKEPU_MAPOVERLAP_INPUT_TYPE> *input, SKEPU_HOST_KERNEL_PARAMS
		skepu2::backend::DeviceMemPointer_CL<SKEPU_MAPOVERLAP_RESULT_TYPE> *output, skepu2::backend::DeviceMemPointer_CL<SKEPU_MAPOVERLAP_INPUT_TYPE> *wrap,
		size_t n, size_t overlap, size_t out_offset, size_t out_numelements, int poly, SKEPU_MAPOVERLAP_INPUT_TYPE pad,
		size_t sharedMemSize
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_VECTOR);
		skepu2::backend::cl_helpers::setKernelArgs(kernel, input->getDeviceDataPointer(), SKEPU_KERNEL_ARGS output->getDeviceDataPointer(),
			wrap->getDeviceDataPointer(), n, overlap, out_offset, out_numelements, poly, pad);
		clSetKernelArg(kernel, SKEPU_KERNEL_ARG_COUNT + 9, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu2::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(), kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching MapOverlap 1D vector kernel");
	}
	
	static void mapOverlapMatrixRowWise
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		skepu2::backend::DeviceMemPointer_CL<SKEPU_MAPOVERLAP_INPUT_TYPE> *input, SKEPU_HOST_KERNEL_PARAMS
		skepu2::backend::DeviceMemPointer_CL<SKEPU_MAPOVERLAP_RESULT_TYPE> *output, skepu2::backend::DeviceMemPointer_CL<SKEPU_MAPOVERLAP_INPUT_TYPE> *wrap,
		size_t n, size_t overlap, size_t out_offset, size_t out_numelements, int poly, SKEPU_MAPOVERLAP_INPUT_TYPE pad, size_t blocksPerRow, size_t rowWidth,
		size_t sharedMemSize
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_MATRIX_ROW);
		skepu2::backend::cl_helpers::setKernelArgs(kernel, input->getDeviceDataPointer(), SKEPU_KERNEL_ARGS output->getDeviceDataPointer(),
			wrap->getDeviceDataPointer(), n, overlap, out_offset, out_numelements, poly, pad, blocksPerRow, rowWidth);
		clSetKernelArg(kernel, SKEPU_KERNEL_ARG_COUNT + 11, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu2::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(),
			kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching MapOverlap 1D matrix row-wise kernel");
	}
	
	static void mapOverlapMatrixColWise
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		skepu2::backend::DeviceMemPointer_CL<SKEPU_MAPOVERLAP_INPUT_TYPE> *input, SKEPU_HOST_KERNEL_PARAMS
		skepu2::backend::DeviceMemPointer_CL<SKEPU_MAPOVERLAP_RESULT_TYPE> *output, skepu2::backend::DeviceMemPointer_CL<SKEPU_MAPOVERLAP_INPUT_TYPE> *wrap,
		size_t n, size_t overlap, size_t out_offset, size_t out_numelements, int poly, SKEPU_MAPOVERLAP_INPUT_TYPE pad, size_t blocksPerCol, size_t rowWidth, size_t colWidth,
		size_t sharedMemSize
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_MATRIX_COL);
		skepu2::backend::cl_helpers::setKernelArgs(kernel, input->getDeviceDataPointer(), SKEPU_KERNEL_ARGS output->getDeviceDataPointer(),
			wrap->getDeviceDataPointer(), n, overlap, out_offset, out_numelements, poly, pad, blocksPerCol, rowWidth, colWidth);
		clSetKernelArg(kernel, SKEPU_KERNEL_ARG_COUNT + 12, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu2::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(),
			kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching MapOverlap 1D matrix col-wise kernel");
	}
	
	static void mapOverlapMatrixColWiseMulti
	(
		size_t deviceID, size_t localSize, size_t globalSize,
		skepu2::backend::DeviceMemPointer_CL<SKEPU_MAPOVERLAP_INPUT_TYPE> *input, SKEPU_HOST_KERNEL_PARAMS
		skepu2::backend::DeviceMemPointer_CL<SKEPU_MAPOVERLAP_RESULT_TYPE> *output, skepu2::backend::DeviceMemPointer_CL<SKEPU_MAPOVERLAP_INPUT_TYPE> *wrap,
		size_t n, size_t overlap, size_t in_offset, size_t out_numelements, int poly, int deviceType, SKEPU_MAPOVERLAP_INPUT_TYPE pad, size_t blocksPerCol, size_t rowWidth, size_t colWidth,
		size_t sharedMemSize
	)
	{
		cl_kernel kernel = kernels(deviceID, KERNEL_MATRIX_COL_MULTI);
		skepu2::backend::cl_helpers::setKernelArgs(kernel, input->getDeviceDataPointer(), SKEPU_KERNEL_ARGS output->getDeviceDataPointer(),
			wrap->getDeviceDataPointer(), n, overlap, in_offset, out_numelements, poly, deviceType, pad, blocksPerCol, rowWidth, colWidth);
		clSetKernelArg(kernel, SKEPU_KERNEL_ARG_COUNT + 13, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu2::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(),
			kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching MapOverlap 1D matrix col-wise multi kernel");
	}
};
)~~~";

std::string generateUserFunctionCode_MapOverlap_CL(UserFunction &Func, bool is2D)
{
	std::stringstream SSFuncParamList;
	std::stringstream SSFuncParams;
	
	SSFuncParamList         << Func.elwiseParams[0].resolvedTypeName << " " << Func.elwiseParams[0].name;
	SSFuncParamList << ", " << Func.elwiseParams[1].resolvedTypeName << " " << Func.elwiseParams[1].name;
	
	if (is2D)
	{
		SSFuncParamList << ", " << Func.elwiseParams[2].resolvedTypeName << " " << Func.elwiseParams[2].name;
		SSFuncParamList << ", __local " << Func.elwiseParams[3].resolvedTypeName << " " << Func.elwiseParams[3].name;
	}
	else
		SSFuncParamList << ", __local " << Func.elwiseParams[2].resolvedTypeName << " " << Func.elwiseParams[2].name;
	
	for (UserFunction::RandomAccessParam& param : Func.anyContainerParams)
		SSFuncParamList << ", " << param.TypeNameOpenCL() << " " << param.name;
	
	for (UserFunction::Param& param : Func.anyScalarParams)
		SSFuncParamList << ", " << param.resolvedTypeName << " " << param.name;
	
	std::string transformedSource = replaceReferencesToOtherUFs(Func, [] (UserFunction &UF) { return UF.uniqueName; });
	
	std::stringstream SSFuncSource;
	
	for (UserFunction *RefFunc : Func.ReferencedUFs)
		SSFuncSource << generateUserFunctionCode_CL(*RefFunc);
	
	SSFuncSource << "static " << Func.resolvedReturnTypeName << " " << Func.uniqueName << "(" << SSFuncParamList.str() << ")\n{";
	for (UserFunction::TemplateArgument &arg : Func.templateArguments)
		SSFuncSource << "typedef " << arg.typeName << " " << arg.paramName << ";\n";
		
	SSFuncSource << transformedSource << "\n}\n\n";
	return SSFuncSource.str();
}

std::string createMapOverlap1DKernelProgram_CL(UserFunction &mapOverlapFunc, std::string dir)
{
	std::stringstream sourceStream, SSMapOverlapFuncArgs, SSKernelParamList, SSHostKernelParamList, SSKernelArgs, SSProxyInitializer;
	std::map<ContainerType, std::set<std::string>> containerProxyTypes;
	
	for (UserFunction::RandomAccessParam& param : mapOverlapFunc.anyContainerParams)
	{
		std::string name = "skepu_container_" + param.name;
		SSHostKernelParamList << param.TypeNameHost() << " skepu_container_" << param.name << ", ";
		containerProxyTypes[param.containerType].insert(param.resolvedTypeName);
		switch (param.containerType)
		{
			case ContainerType::Vector:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", size_t skepu_size_" << param.name << ", ";
				SSKernelArgs << "std::get<1>(" << name << ")->getDeviceDataPointer(), std::get<0>(" << name << ")->size(), ";
				SSProxyInitializer << param.TypeNameOpenCL() << " " << param.name << " = { .data = " << name << ", .size = skepu_size_" << param.name << " };\n";
				break;
			case ContainerType::Matrix:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", size_t skepu_rows_" << param.name << ", size_t skepu_cols_" << param.name << ", ";
				SSKernelArgs << "std::get<1>(" << name << ")->getDeviceDataPointer(), std::get<0>(" << name << ")->total_rows(), std::get<0>(" << name << ")->total_cols(), ";
				SSProxyInitializer << param.TypeNameOpenCL() << " " << param.name << " = { .data = " << name
					<< ", .rows = skepu_rows_" << param.name << ", .cols = skepu_cols_" << param.name << " };\n";
				break;
			case ContainerType::SparseMatrix:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", size_t skepu_size_" << param.name << ", ";
				SSKernelArgs << "skepu_container_" << param.name << ".size(), ";
				break;
		}
		SSMapOverlapFuncArgs << ", " << param.name;
	}
	
	for (UserFunction::Param& param : mapOverlapFunc.anyScalarParams)
	{
		SSKernelParamList << param.resolvedTypeName << " " << param.name << ", ";
		SSHostKernelParamList << param.resolvedTypeName << " " << param.name << ", ";
		SSKernelArgs << param.name << ", ";
		SSMapOverlapFuncArgs << ", " << param.name;
	}
	
	const Type *type = mapOverlapFunc.elwiseParams[2].astDeclNode->getOriginalType().getTypePtr();
	const PointerType *pointer = dyn_cast<PointerType>(type);
	QualType pointee = pointer->getPointeeType().getUnqualifiedType();
	std::string elemType = pointee.getAsString();
	
	// Include user constants as preprocessor macros
	for (auto pair : UserConstants)
		sourceStream << "#define " << pair.second->name << " (" << pair.second->definition << ") // " << pair.second->typeName << "\n";
	
	// check for extra user-supplied opencl code for custome datatype
	for (UserType *RefType : mapOverlapFunc.ReferencedUTs)
		sourceStream << generateUserTypeCode_CL(*RefType);
	
	if (mapOverlapFunc.requiresDoublePrecision)
		sourceStream << "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";
		
	for (const std::string &type : containerProxyTypes[ContainerType::Vector])
		sourceStream << generateOpenCLVectorProxy(type);
	
	for (const std::string &type : containerProxyTypes[ContainerType::Matrix])
		sourceStream << generateOpenCLMatrixProxy(type);
	
	for (const std::string &type : containerProxyTypes[ContainerType::SparseMatrix])
		sourceStream << generateOpenCLSparseMatrixProxy(type);
	
	sourceStream << KernelPredefinedTypes_CL << generateUserFunctionCode_MapOverlap_CL(mapOverlapFunc, false)
	             << MapOverlapKernel_CL << MapOverlapKernel_CL_Matrix_Row
	             << MapOverlapKernel_CL_Matrix_Col << MapOverlapKernel_CL_Matrix_ColMulti;
	
	const std::string kernelName = transformToCXXIdentifier(ResultName) + "_OverlapKernel_" + mapOverlapFunc.uniqueName;
	const std::string className = "CLWrapperClass_" + kernelName;
	std::stringstream SSKernelArgCount;
	SSKernelArgCount << (mapOverlapFunc.numKernelArgsCL() - 3);
	
	std::string finalSource = Constructor1D;
	replaceTextInString(finalSource, "SKEPU_OPENCL_KERNEL", sourceStream.str());
	replaceTextInString(finalSource, PH_MapOverlapInputType, elemType);
	replaceTextInString(finalSource, PH_MapOverlapResultType, mapOverlapFunc.resolvedReturnTypeName);
	replaceTextInString(finalSource, PH_KernelName, kernelName);
	replaceTextInString(finalSource, PH_MapOverlapFuncName, mapOverlapFunc.uniqueName);
	replaceTextInString(finalSource, PH_KernelParams, SSKernelParamList.str());
	replaceTextInString(finalSource, PH_MapOverlapArgs, SSMapOverlapFuncArgs.str());
	replaceTextInString(finalSource, "SKEPU_HOST_KERNEL_PARAMS", SSHostKernelParamList.str());
	replaceTextInString(finalSource, "SKEPU_KERNEL_CLASS", className);
	replaceTextInString(finalSource, "SKEPU_KERNEL_ARGS", SSKernelArgs.str());
	replaceTextInString(finalSource, "SKEPU_KERNEL_ARG_COUNT", SSKernelArgCount.str());
	replaceTextInString(finalSource, "SKEPU_CONTAINER_PROXIES", SSProxyInitializer.str());
	
	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << finalSource;
	
	return kernelName;
}




/*!
* The mapoverlap OpenCL kernel to apply a user function on neighbourhood of each element in the matrix.
*/
static const std::string MatrixConvol2D_CL = R"~~~(
__kernel void SKEPU_KERNEL_NAME(
	__global SKEPU_MAPOVERLAP_INPUT_TYPE* input, SKEPU_KERNEL_PARAMS __global SKEPU_MAPOVERLAP_RESULT_TYPE* output,
	size_t out_rows, size_t out_cols, size_t overlap_y, size_t overlap_x,
	size_t in_pitch, size_t sharedRows, size_t sharedCols,
	__local SKEPU_MAPOVERLAP_INPUT_TYPE* sdata)
{
	size_t xx = ((size_t)(get_global_id(0) / get_local_size(0))) * get_local_size(0);
	size_t yy = ((size_t)(get_global_id(1) / get_local_size(1))) * get_local_size(1);
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);
	SKEPU_CONTAINER_PROXIES

	if (x < out_cols + overlap_x * 2 && y < out_rows + overlap_y * 2)
	{
		size_t sharedIdx = get_local_id(1) * sharedCols + get_local_id(0);
		sdata[sharedIdx]= input[y * in_pitch + x];
		
		size_t shared_x = get_local_id(0)+get_local_size(0);
		size_t shared_y = get_local_id(1);
		while (shared_y < sharedRows)
		{
			while (shared_x < sharedCols)
			{
				sharedIdx = shared_y * sharedCols + shared_x; 
				sdata[sharedIdx] = input[(yy + shared_y) * in_pitch + xx + shared_x];
				shared_x = shared_x + get_local_size(0);
			}
			shared_x = get_local_id(0);
			shared_y = shared_y + get_local_size(1);
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (x < out_cols && y < out_rows)
		output[y * out_cols + x] = SKEPU_FUNCTION_NAME_MAPOVERLAP(overlap_x, overlap_y,
			sharedCols, &sdata[(get_local_id(1) + overlap_y) * sharedCols + (get_local_id(0) + overlap_x)] SKEPU_MAPOVERLAP_ARGS);
}
)~~~";


const std::string Constructor2D = R"~~~(
class SKEPU_KERNEL_CLASS
{
public:
	
	static cl_kernel kernels(size_t deviceID, cl_kernel *newkernel = nullptr)
	{
		static cl_kernel arr[8]; // Hard-coded maximum
		if (newkernel)
		{
			arr[deviceID] = *newkernel;
			return nullptr;
		}
		else return arr[deviceID];
	}
	
	static void initialize()
	{
		static bool initialized = false;
		if (initialized)
			return;
		
		std::string source = skepu2::backend::cl_helpers::replaceSizeT(R"###(SKEPU_OPENCL_KERNEL)###");
		
		// Builds the code and creates kernel for all devices
		size_t counter = 0;
		for (skepu2::backend::Device_CL *device : skepu2::backend::Environment<int>::getInstance()->m_devices_CL)
		{
			cl_int err;
			cl_program program = skepu2::backend::cl_helpers::buildProgram(device, source);
			cl_kernel kernel = clCreateKernel(program, "SKEPU_KERNEL_NAME", &err);
			CL_CHECK_ERROR(err, "Error creating MapOverlap 2D kernel '" << "SKEPU_KERNEL_NAME" << "'");
			
			kernels(counter++, &kernel);
		}
		
		initialized = true;
	}
	
	static void mapOverlap2D
	(
		size_t deviceID, size_t localSize[2], size_t globalSize[2],
		skepu2::backend::DeviceMemPointer_CL<SKEPU_MAPOVERLAP_INPUT_TYPE> *input, SKEPU_HOST_KERNEL_PARAMS
		skepu2::backend::DeviceMemPointer_CL<SKEPU_MAPOVERLAP_RESULT_TYPE> *output,
		size_t out_rows, size_t out_cols, size_t overlap_y, size_t overlap_x,
		size_t in_pitch, size_t sharedRows, size_t sharedCols,
		size_t sharedMemSize
	)
	{
		skepu2::backend::cl_helpers::setKernelArgs(kernels(deviceID), input->getDeviceDataPointer(), SKEPU_KERNEL_ARGS output->getDeviceDataPointer(),
			out_rows, out_cols, overlap_y, overlap_x, in_pitch, sharedRows, sharedCols);
		clSetKernelArg(kernels(deviceID), SKEPU_KERNEL_ARG_COUNT + 9, sharedMemSize, NULL);
		cl_int err = clEnqueueNDRangeKernel(skepu2::backend::Environment<int>::getInstance()->m_devices_CL.at(deviceID)->getQueue(),
			kernels(deviceID), 2, NULL, globalSize, localSize, 0, NULL, NULL);
		CL_CHECK_ERROR(err, "Error launching MapOverlap 2D kernel");
	}
};
)~~~";

std::string createMapOverlap2DKernelProgram_CL(UserFunction &mapOverlapFunc, std::string dir)
{
	std::stringstream sourceStream, SSMapOverlapFuncArgs, SSKernelParamList, SSHostKernelParamList, SSKernelArgs, SSProxyInitializer;
	std::map<ContainerType, std::set<std::string>> containerProxyTypes;
	
	for (UserFunction::RandomAccessParam& param : mapOverlapFunc.anyContainerParams)
	{
		std::string name = "skepu_container_" + param.name;
		SSHostKernelParamList << param.TypeNameHost() << " skepu_container_" << param.name << ", ";
		containerProxyTypes[param.containerType].insert(param.resolvedTypeName);
		switch (param.containerType)
		{
			case ContainerType::Vector:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", size_t skepu_size_" << param.name << ", ";
				SSKernelArgs << "std::get<1>(" << name << ")->getDeviceDataPointer(), std::get<0>(" << name << ")->size(), ";
				SSProxyInitializer << param.TypeNameOpenCL() << " " << param.name << " = { .data = " << name << ", .size = skepu_size_" << param.name << " };\n";
				break;
			case ContainerType::Matrix:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", size_t skepu_rows_" << param.name << ", size_t skepu_cols_" << param.name << ", ";
				SSKernelArgs << "std::get<1>(" << name << ")->getDeviceDataPointer(), std::get<0>(" << name << ")->total_rows(), std::get<0>(" << name << ")->total_cols(), ";
				SSProxyInitializer << param.TypeNameOpenCL() << " " << param.name << " = { .data = " << name
					<< ", .rows = skepu_rows_" << param.name << ", .cols = skepu_cols_" << param.name << " };\n";
				break;
			case ContainerType::SparseMatrix:
				SSKernelParamList << "__global " << param.resolvedTypeName << " *" << name << ", size_t skepu_size_" << param.name << ", ";
				SSKernelArgs << "skepu_container_" << param.name << ".size(), ";
				break;
		}
		SSMapOverlapFuncArgs << ", " << param.name;
	}
	
	for (UserFunction::Param& param : mapOverlapFunc.anyScalarParams)
	{
		SSKernelParamList << param.resolvedTypeName << " " << param.name << ", ";
		SSHostKernelParamList << param.resolvedTypeName << " " << param.name << ", ";
		SSKernelArgs << param.name << ", ";
		SSMapOverlapFuncArgs << ", " << param.name;
	}
	
	const Type *type = mapOverlapFunc.elwiseParams[3].astDeclNode->getOriginalType().getTypePtr();
	const PointerType *pointer = dyn_cast<PointerType>(type);
	QualType pointee = pointer->getPointeeType().getUnqualifiedType();
	std::string elemType = pointee.getAsString();
	
	// Include user constants as preprocessor macros
	for (auto pair : UserConstants)
		sourceStream << "#define " << pair.second->name << " (" << pair.second->definition << ") // " << pair.second->typeName << "\n";
	
	// check for extra user-supplied opencl code for custome datatype
	for (UserType *RefType : mapOverlapFunc.ReferencedUTs)
		sourceStream << generateUserTypeCode_CL(*RefType);
	
	if (mapOverlapFunc.requiresDoublePrecision)
		sourceStream << "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n";
	
	for (const std::string &type : containerProxyTypes[ContainerType::Vector])
		sourceStream << generateOpenCLVectorProxy(type);
	
	for (const std::string &type : containerProxyTypes[ContainerType::Matrix])
		sourceStream << generateOpenCLMatrixProxy(type);
	
	for (const std::string &type : containerProxyTypes[ContainerType::SparseMatrix])
		sourceStream << generateOpenCLSparseMatrixProxy(type);
	
	sourceStream << KernelPredefinedTypes_CL << generateUserFunctionCode_MapOverlap_CL(mapOverlapFunc, true) << MatrixConvol2D_CL;
	
	const std::string kernelName = transformToCXXIdentifier(ResultName) + "_Overlap2DKernel_" + mapOverlapFunc.uniqueName;
	const std::string className = "CLWrapperClass_" + kernelName;
	std::stringstream SSKernelArgCount;
	SSKernelArgCount << (mapOverlapFunc.numKernelArgsCL() - 4);
	
	std::string finalSource = Constructor2D;
	replaceTextInString(finalSource, "SKEPU_OPENCL_KERNEL", sourceStream.str());
	replaceTextInString(finalSource, PH_MapOverlapInputType, elemType);
	replaceTextInString(finalSource, PH_MapOverlapResultType, mapOverlapFunc.resolvedReturnTypeName);
	replaceTextInString(finalSource, PH_KernelName, kernelName);
	replaceTextInString(finalSource, PH_MapOverlapFuncName, mapOverlapFunc.uniqueName);
	replaceTextInString(finalSource, PH_KernelParams, SSKernelParamList.str());
	replaceTextInString(finalSource, PH_MapOverlapArgs, SSMapOverlapFuncArgs.str());
	replaceTextInString(finalSource, "SKEPU_HOST_KERNEL_PARAMS", SSHostKernelParamList.str());
	replaceTextInString(finalSource, "SKEPU_KERNEL_CLASS", className);
	replaceTextInString(finalSource, "SKEPU_KERNEL_ARGS", SSKernelArgs.str());
	replaceTextInString(finalSource, "SKEPU_KERNEL_ARG_COUNT", SSKernelArgCount.str());
	replaceTextInString(finalSource, "SKEPU_CONTAINER_PROXIES", SSProxyInitializer.str());
	
	std::ofstream FSOutFile {dir + "/" + kernelName + "_cl_source.inl"};
	FSOutFile << finalSource;
	
	return kernelName;
}
