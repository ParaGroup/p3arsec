/*!
Mandelbrot fractals. The Mandelbrot set
{
"B. B. Mandelbrot. Fractal aspects of the iteration of z → λz(1 − z) for complex λ and z.
Annals of the New York Academy of Sciences, 357:249–259, December 1980."
}
is a set of complex numbers which boundary draws a fractal in the complex numbers plane. A complex number c lies
within the Mandelbrot set, if the sequence
z_{i+1} = z_{i}2 + c
with i ∈ N, starting with z0 = 0 does not escape to infinity, otherwise c is not part of
the Mandelbrot set.

When computing a Mandelbrot fractal, the sequence in equation 3.1 is calculated
for every pixel of an image representing a section of the complex numbers plane. If a
given threshold is crossed, it is presumed that the sequence will escape to infinity and
that the pixel is not inside the Mandelbrot set. If the threshold is not crossed for a given
number of steps in the sequence, the pixel is taken as a member of the Mandelbrot
set. A pixel within the Mandelbrot set painted in black, other pixels are given a color
that corresponds to the number of sequence steps that have been calculated before
excluding the pixel from the Mandelbrot set. By setting the threshold and the number
of sequence steps accordingly, the calculation of the fractal can be a time-consuming
task. However, as all pixels are calculated independently, it is a common benchmark
application for data-parallel computations.
*/

#include <iostream>
#include <fstream>
#include <cstdlib>

#include <skepu2.hpp>

#include "bitmap_image.hpp"

template<typename T>
void save_image(size_t width, size_t height, T *buf, T max)
{
	bitmap_image image(width, height);
	
	for (size_t y = 0; y < height; ++y)
	{
		for (size_t x = 0; x < width; ++x)
		{
			float val = buf[y*width + x];
			unsigned char shade = val / max * 255;
			rgb_store col = prism_colormap[shade];
			image.set_pixel(x, y, col.red, col.green, col.blue);
		}
	}
	
	image.save_image("generated_image.bmp");
}


[[skepu::userconstant]] constexpr float
	CENTER_X = -.5f,
	CENTER_Y = 0.f,
	SCALE = 2.5f;

[[skepu::userconstant]] constexpr size_t
	MAX_ITERS = 1000;

struct cplx 
{
	float a, b;
};

cplx mult_c(cplx lhs, cplx rhs)
{
	cplx r;
	r.a = lhs.a * rhs.a - lhs.b * rhs.b;
	r.b = lhs.b * rhs.a + lhs.a * rhs.b;
	return r;
}

cplx add_c(cplx lhs, cplx rhs)
{
	cplx r;
	r.a = lhs.a + rhs.a;
	r.b = lhs.b + rhs.b;
	return r;
}

size_t mandelbrot_f(skepu2::Index2D index, size_t height, size_t width)
{
	cplx a;
	a.a = SCALE / height * (index.col - width/2.f) + CENTER_X;
	a.b = SCALE / height * (index.row - width/2.f) + CENTER_Y;
	cplx c = a;
	
	for (size_t i = 0; i < MAX_ITERS; ++i)
	{
		a = add_c(mult_c(a, a), c);
		if ((a.a * a.a + a.b * a.b) > 4)
			return i;
	}
	return MAX_ITERS;
}



auto mandelbroter = skepu2::Map<0>(mandelbrot_f);

void mandelbrot(skepu2::Matrix<size_t> &iterations, skepu2::BackendSpec *spec = nullptr)
{
	const size_t width = iterations.total_cols();
	const size_t height = iterations.total_rows();
	
	
	if (spec)
		mandelbroter.setBackend(*spec);
	
	mandelbroter(iterations, height, width);
}

int main(int argc, char* argv[])
{
	if (argc < 4)
	{
		std::cout << "Usage: " << argv[0] << " width height backend\n";
		exit(1);
	}
	
	const size_t width = std::stoul(argv[1]);
	const size_t height = std::stoul(argv[2]);
	auto spec = skepu2::BackendSpec{skepu2::Backend::typeFromString(argv[3])};
	
	skepu2::Matrix<size_t> iterations(height, width);
	
	mandelbrot(iterations, &spec);
	iterations.updateHost();
	
	save_image(width, height, iterations.getAddress(), MAX_ITERS);
}
