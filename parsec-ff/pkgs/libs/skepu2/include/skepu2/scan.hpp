#pragma once

#include "skepu2/impl/common.hpp"

namespace skepu2
{
	enum class ScanMode
	{
		Inclusive, Exclusive
	};
	
	namespace impl
	{
		template<typename>
		class ScanImpl;
	}
	
	template<typename T>
	impl::ScanImpl<T> ScanWrapper(std::function<T(T, T)> scan)
	{
		return impl::ScanImpl<T>(scan);
	}
	
	// For function pointers
	template<typename T>
	impl::ScanImpl<T> Scan(T(*scan)(T, T))
	{
		return ScanWrapper((std::function<T(T, T)>)scan);
	}
	
	// For lambdas and functors
	template<typename T>
	auto Scan(T scan) -> decltype(ScanWrapper(lambda_cast(scan)))
	{
		return ScanWrapper(lambda_cast(scan));
	}
	
	namespace impl
	{
		template<typename T>
		class ScanImpl: public SeqSkeletonBase
		{
			using ScanFunc = std::function<T(T, T)>;
		
			ScanMode m_mode {ScanMode::Inclusive};
			
			// Default initial value is a default-initialized value
			T m_initial {};
			
			ScanFunc m_scanFunc;
			
			ScanImpl(ScanFunc scan): m_scanFunc(scan) {}
			
			
			
			template<typename OutIterator, typename InIterator>
			void apply(size_t size, OutIterator res, InIterator arg)
			{
				T running = this->m_initial;
				
				if (size != res.size())
					SKEPU_ERROR("Non-matching container sizes");
				
				// First element
				if (this->m_mode == ScanMode::Inclusive)
					running = *arg++;
				*res++ = running;
				
				// Rest of elements
				while (--size > 0)
					*res++ = running = this->m_scanFunc(running, *arg++);
			}
			
			friend impl::ScanImpl<T> skepu2::ScanWrapper<T>(ScanFunc);
			
		public:
			
			void setScanMode(ScanMode mode)
			{
				this->m_mode = mode;
			}
			
			void setStartValue(T initial)
			{
				this->m_initial = initial;
			}
			
			template<typename OutIterator, typename In>
			OutIterator operator()(OutIterator res, OutIterator res_end, In&& arg)
			{
				this->apply(res_end - res, res, arg.begin());
				return res;
			}
			
			template<template<class> class Container, typename In>
			Container<T> &operator()(Container<T>& res, In&& arg)
			{
				this->apply(res.size(), res.begin(), arg.begin());
				return res;
			}
			
		};
		
	}
	
}
