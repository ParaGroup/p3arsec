#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <time.h>
#include <iterator>

#include <skepu2.hpp>
#include "lodepng.h"

// Information about the image. Used to rebuild the image after filtering.
struct ImageInfo
{
	int width;
	int height;
	int elementsPerPixel;
};

// Reads a file from png and retuns it as a skepu::Matrix. Uses a library called LodePNG.
// Also returns information about the image as an out variable in imageInformation.
skepu2::Matrix<unsigned char> ReadAndPadPngFileToMatrix(std::string filePath, int kernelRadius, LodePNGColorType colorType, ImageInfo& imageInformation)
{
	std::vector<unsigned char> fileContents, image;
	unsigned imageWidth, imageHeight;
	unsigned error = lodepng::decode(image, imageWidth, imageHeight, filePath, colorType);
	if (error)
		std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
	
	int elementsPerPixel = (colorType == LCT_GREY) ? 1 : 3;
	
	// Create a matrix which fits the image and the padding needed.
	skepu2::Matrix<unsigned char> inputMatrix(imageHeight + 2*kernelRadius, (imageWidth + 2*kernelRadius) * elementsPerPixel);
	
	int nonEdgeStartX = kernelRadius * elementsPerPixel;
	int nonEdgeEndX = inputMatrix.total_cols() - kernelRadius * elementsPerPixel;
	int nonEdgeStartY = kernelRadius;
	int nonEdgeEndY = inputMatrix.total_rows() - kernelRadius;
	
	// Initialize the inner real image values. The image is placed in the middle of the matrix, 
	// surrounded by the padding.
	for (int i = nonEdgeStartY; i < nonEdgeEndY; i++)
		for (int j = nonEdgeStartX; j < nonEdgeEndX; j++)
			inputMatrix(i, j)= image[(i - nonEdgeStartY) * imageWidth * elementsPerPixel + (j - nonEdgeStartX)];
	
	// Initialize padding. // Init topmost rows
	for (int row = 0;row < kernelRadius; row++)
	{
		for (int col = 0; col < inputMatrix.total_cols(); col++)
		{
			int minClampEdgeX = nonEdgeStartX + col % elementsPerPixel; 
			int maxClampEdgeX = nonEdgeEndX - elementsPerPixel + col % elementsPerPixel; 
			int xIndex = std::min(maxClampEdgeX, std::max(col, minClampEdgeX));
			int yIndex = std::min(nonEdgeEndY - 1, std::max(row, nonEdgeStartY));
			inputMatrix(row, col) = inputMatrix(yIndex, xIndex);
		}
	}
	
	// Init middle rows
	for (int row = kernelRadius; row < nonEdgeEndY; row++)
	{
		for (int col = 0; col < nonEdgeStartX; col++)
		{
			int minClampEdgeX = nonEdgeStartX + col % elementsPerPixel; 
			int maxClampEdgeX = nonEdgeEndX - elementsPerPixel + col % elementsPerPixel; 
			inputMatrix(row, col) = inputMatrix(row, minClampEdgeX);
			inputMatrix(row, col + nonEdgeEndX) = inputMatrix(row, maxClampEdgeX);
		}
	}
	
	// Init bottom rows
	for (int row = nonEdgeEndY; row < inputMatrix.total_rows(); row++)
	{
		for (int col = 0; col < inputMatrix.total_cols(); col++)
		{
			int minClampEdgeX = nonEdgeStartX + col % elementsPerPixel; 
			int maxClampEdgeX = nonEdgeEndX - elementsPerPixel + col % elementsPerPixel; 
			int xIndex = std::min(maxClampEdgeX, std::max(col, minClampEdgeX));
			int yIndex = std::min(nonEdgeEndY - 1, std::max(row, nonEdgeStartY));
			inputMatrix(row, col) = inputMatrix(yIndex, xIndex);
		}
	}
	
	imageInformation.height = imageHeight;
	imageInformation.width = imageWidth;
	imageInformation.elementsPerPixel = elementsPerPixel;
	return inputMatrix;
}

void WritePngFileMatrix(skepu2::Matrix<unsigned char> imageData, std::string filePath, LodePNGColorType colorType, ImageInfo& imageInformation)
{
	std::vector<unsigned char> imageDataVector; 
	for (int i = 0; i < imageData.total_rows(); i++)
		for (int j = 0; j < imageData.total_cols() ;j++)
			imageDataVector.push_back(imageData(i, j));
	
	unsigned error = lodepng::encode(filePath, &imageDataVector[0], imageInformation.width, imageInformation.height, colorType);
	if(error)
		std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
}




// Kernel for filter with raduis R
unsigned char median_kernel(int ox, int oy, size_t stride, const unsigned char *image, size_t elemPerPx)
{
	long fineHistogram[256], coarseHistogram[16];
	
	for (int i = 0; i < 256; i++)
		fineHistogram[i] = 0;
	
	for (int i = 0; i < 16; i++)
		coarseHistogram[i] = 0;
	
	for (int row = -oy; row <= oy; row++)
	{
		for (int column = -ox; column <= ox; column += elemPerPx)
		{ 
			unsigned char imageValue = image[row * stride + column];
			fineHistogram[imageValue]++;
			coarseHistogram[imageValue / 16]++;
		}
	}
	
	int count = 2 * oy * (oy + 1);
	
	unsigned char coarseIndex;
	for (coarseIndex = 0; coarseIndex < 16; ++coarseIndex)
	{
		if ((long)count - coarseHistogram[coarseIndex] < 0) break;
		count -= coarseHistogram[coarseIndex];
	}
	
//	int coarseIndex = 0;
//	while (count - coarseHistogram[coarseIndex] >= 0)
//		count -= coarseHistogram[coarseIndex++];
	
	unsigned char fineIndex = coarseIndex * 16;
	while ((long)count - fineHistogram[fineIndex] >= 0)
		count -= fineHistogram[fineIndex++];
	
	return fineIndex;
}

int main(int argc, char* argv[])
{
	LodePNGColorType colorType = LCT_RGB;
	
	if (argc < 3)
	{
		std::cout << "Usage: " << argv[0] << "input-image output-image start-radius increment-radius count [backend]\n";
		exit(1);
	}
	
	std::string inputFileName = argv[1];
	std::string outputFileName = argv[2];
	skepu2::BackendSpec spec;
	if (argc > 3) spec = skepu2::BackendSpec{skepu2::Backend::typeFromString(argv[6])};
	
	auto calculateMedian = skepu2::MapOverlap(median_kernel);
	calculateMedian.setBackend(spec);
	
	const int startRadius = atoi(argv[3]);
	const int incrementRadius = atoi(argv[4]);
	const int finalRadius = startRadius + incrementRadius * atoi(argv[5]);
	
	// Loop through all different kernel radiuses we will test.
	for (int kernelRadius = startRadius; kernelRadius < finalRadius; kernelRadius += incrementRadius)
	{
		// Create the full path for writing the image.
		std::stringstream ss;
		ss << (2 * kernelRadius + 1) << "x" << (2 * kernelRadius + 1);
		std::string outputFileNamePad = outputFileName + ss.str() + ".png";
		
		// Read the padded image into a matrix. Create the output matrix without padding.
		ImageInfo imageInfo;
		skepu2::Matrix<unsigned char> inputMatrix = ReadAndPadPngFileToMatrix(inputFileName, kernelRadius, colorType, imageInfo);
		skepu2::Matrix<unsigned char> outputMatrix(imageInfo.height, imageInfo.width * imageInfo.elementsPerPixel, 120);
		
		int testRuns = 1;
		double timeTaken = 0;
		
		// Run the filtering multiple times using the same kernel and average the values.
		for (int j = 0; j < testRuns; j++)
		{
			// Launch different kernels depending on filtersize.
			calculateMedian.setOverlap(kernelRadius, kernelRadius  * imageInfo.elementsPerPixel);
			clock_t t1 = clock();
			calculateMedian(outputMatrix, inputMatrix, imageInfo.elementsPerPixel);
			
			// The kernel is run with lazy evaluation, so we need to access the output matrix
			outputMatrix.updateHost();
			
			clock_t t2 = clock();
			timeTaken += t2 - t1;
		}
		
		timeTaken /= CLOCKS_PER_SEC;
		timeTaken /= testRuns;
		timeTaken *= 1000;
		std::cout << "--------------------------------------------------------\n";
		std::cout << "Inputfile : " << inputFileName << std::endl;
		std::cout << "Filtersize : " << (2 * kernelRadius + 1) << "x" << (2 * kernelRadius + 1) << std::endl;
		std::cout << "Time taken : " << timeTaken << " ms" << std::endl;
		
		WritePngFileMatrix(outputMatrix, outputFileNamePad, colorType, imageInfo);
	}
	
	return 0;
}
