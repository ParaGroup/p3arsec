#include <iostream>
#include <cmath>

#include <skepu2.hpp>

#define RAND_MAX_LOC 100

typedef struct _dim3
{
	size_t x;
	size_t y;
	size_t z;
} dimension3;

struct Atom
{
	float charge;
	float x, y, z;
};

std::ostream &operator<<(std::ostream &o, Atom &a)
{
	o << "Atom: [ x: " << a.x << " y: " << a.y << " z: " << a.z << " charge: " << a.charge << " ]\n";
	return o;
}


// ***********************
// ***********************
// Please note down the difference that here we use "sqrt" function instead of "sqrtf" function used in CUDA code
// ***********************
// ***********************

// This kernel calculates coulombic potential at each grid point and
// stores the results in the output array.
// Note: this implementation uses precomputed and unrolled
// loops of dy*dy + dz*dz values for increased FP arithmetic intensity
// per FP load.  The X coordinate portion of the loop is unrolled by
// four, allowing the same dy^2 + dz^2 values to be reused four times,
// increasing the ratio of FP arithmetic relative to FP loads, and
// eliminating some redundant calculations.
// This version implement's Kahan's compensated summation method to
// increase doubleing point accuracy for the large number of summed
// potential values.
//
//
float coulombPotential_f(skepu2::Index2D index, float energygridItem, const skepu2::Vec<Atom> atoms, float gridspacing)
{
	float coory = gridspacing * index.col;
	float coorx = gridspacing * index.row;
	
//	std::cout << "X: " << coorx << " Y: " << coory << "\n";
	
	float energyvalx1 = 0;
	float energycomp1 = 0;
	size_t numatoms = atoms.size;
	for (size_t atomid = 0; atomid < numatoms; atomid ++)
	{
		float dy = coory - atoms.data[atomid].y;
		float dx = coorx - atoms.data[atomid].x;
		float s = atoms.data[atomid].charge / sqrt(dx*dx + dy*dy);
		
	//	std::cout << "s: " << s << "\n";
		
		float y = s - energycomp1;
		float t = energyvalx1 + y;
		energycomp1 = (t - energyvalx1)  - y;
		energyvalx1 = t;
	}
	return energygridItem + energyvalx1;
}


// Function to initialize atoms
void initatoms(skepu2::Vector<Atom> &atombuf, dimension3 volsize, double gridspacing)
{
	srand(0);
	
	// compute grid dimensions in angstroms
	dimension3 size;
	size.x = gridspacing * volsize.x;
	size.y = gridspacing * volsize.y;
	size.z = gridspacing * volsize.z;
	
	size_t count = atombuf.size();
	for (size_t i = 0; i < count; i++)
	{
		Atom a;
		a.charge = ((rand() / (double) RAND_MAX) * 2.0) - 1.0;  // charge
		a.x = (rand() / (double) RAND_MAX);// * size.x;
		a.y = (rand() / (double) RAND_MAX);// * size.y;
		a.z = (rand() / (double) RAND_MAX);// * size.z;
		atombuf[i] = a;
	}
}


const float gridspacing = 0.1;
const size_t matrixSize = 10;

auto columbicPotential = skepu2::Map<1>(coulombPotential_f);

void coulombic(skepu2::Matrix<float> &energy_out, skepu2::Vector<Atom> &atombuf, skepu2::BackendSpec *spec = nullptr)
{
	size_t atomcount = atombuf.size();
	
	if (spec)
		columbicPotential.setBackend(*spec);
		
	skepu2::Matrix<float> grid_in(matrixSize, matrixSize);
	
	columbicPotential(energy_out, grid_in, atombuf, gridspacing);
}

int main(int argc, char* argv[])
{
	if (argc < 3)
	{
		std::cout << "Usage: " << argv[0] << " input_size backend\n";
		exit(1);
	}
	
	const size_t atomcount = std::stoul(argv[1]);
	auto spec = skepu2::BackendSpec{skepu2::Backend::typeFromString(argv[2])};
	
	skepu2::Vector<Atom> atoms(atomcount);
	skepu2::Matrix<float> energy_out(matrixSize, matrixSize);
	
	// allocate and initialize atom coordinates and charges
	dimension3 volsize {2048, 2048, 1};
	initatoms(atoms, volsize, gridspacing);
	
	coulombic(energy_out, atoms, &spec);
	
	// can print and compare. output is exactly same for cpu, openmp, cuda for 1 and 2 gpus.
	std::cout << "Energy out: " << energy_out << "\n";
	
	return 0;
}
