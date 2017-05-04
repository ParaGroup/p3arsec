#pragma once

#include "skepu2/impl/common.hpp"

namespace skepu2
{
	enum class ReduceMode
	{
		RowWise, ColWise
	};
	
	inline std::ostream &operator<<(std::ostream &o, ReduceMode m)
	{
		switch (m)
		{
		case ReduceMode::RowWise:
			o << "Rowwise"; break;
		case ReduceMode::ColWise:
			o << "Colwise"; break;
		default:
			o << "<Invalid reduce mode>";
		}
		return o;
	}
	
	namespace impl
	{
		template<typename>
		class Reduce1D;
		
		template<typename>
		class Reduce2D;
	}
	
	
	template<typename T>
	impl::Reduce1D<T> ReduceWrapper(std::function<T(T, T)> red)
	{
		return impl::Reduce1D<T>(red);
	}
	
	// For function pointers
	template<typename T>
	impl::Reduce1D<T> Reduce(T(*red)(T, T))
	{
		return ReduceWrapper((std::function<T(T, T)>)red);
	}
	
	// For lambdas and functors
	template<typename T>
	auto Reduce(T red) -> decltype(ReduceWrapper(lambda_cast(red)))
	{
		return ReduceWrapper(lambda_cast(red));
	}
	
	
	
	template<typename T>
	impl::Reduce2D<T> ReduceWrapper(std::function<T(T, T)> rowRed, std::function<T(T, T)> colRed)
	{
		return impl::Reduce2D<T>(rowRed, colRed);
	}
	
	// For function pointers
	template<typename T>
	impl::Reduce2D<T> Reduce(T(*rowRed)(T, T), T(*colRed)(T, T))
	{
		return ReduceWrapper((std::function<T(T, T)>)rowRed, (std::function<T(T, T)>)colRed);
	}
	
	// For lambdas and functors
	template<typename T1, typename T2>
	auto Reduce(T1 rowRed, T2 colRed) -> decltype(ReduceWrapper(lambda_cast(rowRed), lambda_cast(colRed)))
	{
		return ReduceWrapper(lambda_cast(rowRed), lambda_cast(colRed));
	}
	
	
	namespace impl
	{
		// Reduce1D for Vectors or Matrices
		template<typename T>
		class Reduce1D: public SeqSkeletonBase
		{
			using RedFunc = std::function<T(T, T)>;
			
		public:
			
			void setReduceMode(ReduceMode mode)
			{
				this->m_mode = mode;
			}
			
			void setStartValue(T val)
			{
				this->m_start = val;
			}
			
			
			template<template<class> class Container>
			typename std::enable_if<is_skepu_container<Container<T>>::value, T>::type
			operator()(Container<T>& arg)
			{
				size_t size = arg.size();
				
				T res = this->m_start;
				
				res = arg[0];
				for (size_t i = 1; i < size; i++)
					res = this->redFunc(res, arg[i]);
				
				return res;
			}
			
			Vector<T> &operator()(Vector<T> &res, Matrix<T>& arg)
			{
				size_t rows = arg.total_rows();
				size_t cols = arg.total_cols();
				
				if (this->m_mode == ReduceMode::RowWise)
				{
					if (res.size() != rows)
						SKEPU_ERROR("Reduce: Non-matching container sizes");
					
					for (size_t r = 0; r < rows; r++)
					{
						T inner = arg(r, 0);
						for (size_t c = 1; c < cols; c++)
							inner = this->redFunc(inner, arg(r, c));
						res[r] = inner;
					}
				}
				else if (this->m_mode == ReduceMode::ColWise)
				{
					if (res.size() != cols)
						SKEPU_ERROR("Reduce: Non-matching container sizes");
					
					for (size_t c = 0; c < cols; c++)
					{
						T inner = arg(0, c);
						for (size_t r = 1; r < rows; r++)
							inner = this->redFunc(inner, arg(r, c));
						res[c] = inner;
					}
				}
				
				return res;
			}
			
		protected:
			RedFunc redFunc;
			Reduce1D(RedFunc red): redFunc(red) {}
			
			ReduceMode m_mode = ReduceMode::RowWise;
			T m_start{};
			
			friend Reduce1D<T> ReduceWrapper<T>(RedFunc);
		};
		
		
		// Reduce 2D only for Matrix containers
		// Should be some way to determine order?
		template<typename T>
		class Reduce2D: public Reduce1D<T>
		{
			using RedFunc = std::function<T(T, T)>;
			
		public:
			
			T operator()(Vector<T>& arg)
			{
				return Reduce1D<T>::operator()(arg);
			}
			
			T operator()(Matrix<T>& arg)
			{
				size_t rows = arg.total_rows();
				size_t cols = arg.total_cols();
				
				T res = this->m_start;
				
				if (this->m_mode == ReduceMode::RowWise)
				{
					for (size_t r = 0; r < rows; r++)
					{
						T inner = arg(r, 0);
						for (size_t c = 1; c < cols; c++)
							inner = this->redFunc(inner, arg(r, c));
						res = this->colRedFunc(res, inner);
					}
				}
				else
				{
					for (size_t c = 0; c < cols; c++)
					{
						T inner = arg(0, c);
						for (size_t r = 1; r < rows; r++)
							inner = this->redFunc(inner, arg(r, c));
						res = this->colRedFunc(res, inner);
					}
				}
				
				return res;
			}
			
		private:
			RedFunc colRedFunc;
			Reduce2D(RedFunc rowRed, RedFunc colRed) : Reduce1D<T>(rowRed), colRedFunc(colRed) {}
			
			T m_start{};
			
			friend Reduce2D<T> ReduceWrapper<T>(RedFunc, RedFunc);
		};
	}
	
}
