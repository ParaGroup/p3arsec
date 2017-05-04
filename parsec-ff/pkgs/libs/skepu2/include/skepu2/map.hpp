#pragma once

#include "skepu2/impl/common.hpp"

namespace skepu2
{
	template<int, typename, typename...>
	class MapImpl;
	
	template<int arity = 1, typename Ret, typename... Args>
	MapImpl<arity, Ret, Args...> MapWrapper(std::function<Ret(Args...)> map)
	{
		return MapImpl<arity, Ret, Args...>(map);
	}
	
	// For function pointers
	template<int arity = 1, typename Ret, typename... Args>
	MapImpl<arity, Ret, Args...> Map(Ret(*map)(Args...))
	{
		return MapWrapper<arity>((std::function<Ret(Args...)>)map);
	}
	
	// For lambdas and functors
	template<int arity = 1, typename T>
	auto Map(T map) -> decltype(MapWrapper<arity>(lambda_cast(map)))
	{
		return MapWrapper<arity>(lambda_cast(map));
	}
	
	template<int arity, typename Ret, typename... Args>
	class MapImpl: public SeqSkeletonBase
	{
		static constexpr bool indexed = is_indexed<Args...>::value;
		static constexpr size_t numArgs = sizeof...(Args) - (indexed ? 1 : 0);
		static constexpr size_t anyCont = trait_count_all<is_skepu_container_proxy, Args...>::value;
		
		// Supports the "index trick": using a variant of tag dispatching to index template parameter packs
		static constexpr typename make_pack_indices<arity, 0>::type elwise_indices{};
		static constexpr typename make_pack_indices<arity + anyCont, arity>::type any_indices{};
		static constexpr typename make_pack_indices<numArgs, arity + anyCont>::type const_indices{};
		
		using MapFunc = std::function<Ret(Args...)>;
		using F = ConditionalIndexForwarder<indexed, MapFunc>;
		
		// For iterators
		template<size_t... EI, size_t... AI, size_t... CI, typename Iterator, typename... CallArgs> 
		void apply(pack_indices<EI...>, pack_indices<AI...>, pack_indices<CI...>, size_t size, Iterator res, CallArgs&&... args)
		{
			if (disjunction((get<EI>(args...).size() < size)...))
				SKEPU_ERROR("Non-matching container sizes");
			
			auto elwiseIterators = std::make_tuple(get<EI>(args...).begin()...);
			
			while (size --> 0)
			{
				auto index = res.getIndex();
				*res++ = F::forward(mapFunc, index, *std::get<EI>(elwiseIterators)++..., get<AI>(args...).hostProxy()..., get<CI>(args...)...);
			}
		}
		
		
	public:
		
		void setDefaultSize(size_t x, size_t y = 0)
		{
			this->default_size_x = x;
			this->default_size_y = y;
		}
		
		template<template<class> class Container, typename... CallArgs>
		Container<Ret> &operator()(Container<Ret> &res, CallArgs&&... args)
		{
			static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching Map function");
			this->apply(elwise_indices, any_indices, const_indices, res.size(), res.begin(), std::forward<CallArgs>(args)...);
			return res;
		}
		
		template<typename Iterator, typename... CallArgs, REQUIRES(is_skepu_iterator<Iterator, Ret>())>
		Iterator operator()(Iterator res, Iterator res_end, CallArgs&&... args)
		{
			static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching Map function");
			this->apply(elwise_indices, any_indices, const_indices, res_end - res, res, std::forward<CallArgs>(args)...);
			return res;
		}
		
		template<template<class> class Container = Vector, typename... CallArgs>
		Container<Ret> operator()(CallArgs&&... args)
		{
			static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching Map function");
			
		/*	if (this->default_size_y != 0)
			{
				Container<Ret> res(this->default_size_x, this->default_size_y);
				this->apply(elwise_indices, any_indices, const_indices, res, std::forward<CallArgs>(args)...);
				return std::move(res);
			}
			else
			{*/
				Container<Ret> res(this->default_size_x);
				this->apply(elwise_indices, any_indices, const_indices, res.size(), res.begin(), std::forward<CallArgs>(args)...);
				return std::move(res);
		//	}
		}
		
	private:
		MapFunc mapFunc;
		MapImpl(MapFunc map): mapFunc(map) {}
		
		size_t default_size_x;
		size_t default_size_y;
		
		friend MapImpl<arity, Ret, Args...> MapWrapper<arity, Ret, Args...>(MapFunc);
		
	}; // end class MapImpl
	
} // end namespace skepu2
