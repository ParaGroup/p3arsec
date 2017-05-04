#pragma once

#include "skepu2/impl/common.hpp"

namespace skepu2
{
	template<typename, typename...>
	class CallImpl;
	
	template<typename Ret, typename... Args>
	CallImpl<Ret, Args...> CallWrapper(std::function<Ret(Args...)> call)
	{
		return CallImpl<Ret, Args...>(call);
	}
	
	// For function pointers
	template<typename Ret, typename... Args>
	CallImpl<Ret, Args...> Call(Ret(*call)(Args...))
	{
		return CallWrapper((std::function<Ret(Args...)>)call);
	}
	
	// For lambdas and functors
	template<typename T>
	auto Call(T call) -> decltype(CallWrapper(lambda_cast(call)))
	{
		return CallWrapper(lambda_cast(call));
	}

	template<typename Ret, typename... Args>
	class CallImpl: public SeqSkeletonBase
	{
		static constexpr size_t numArgs = sizeof...(Args);
		static constexpr size_t anyCont = trait_count_all<is_skepu_container_proxy, Args...>::value;
		
		// Supports the "index trick": using a variant of tag dispatching to index template parameter packs
		static constexpr typename make_pack_indices<anyCont, 0>::type any_indices{};
		static constexpr typename make_pack_indices<numArgs, anyCont>::type const_indices{};
		
		using CallFunc = std::function<Ret(Args...)>;
		
		template<size_t... AI, size_t... CI, typename... CallArgs> 
		Ret apply(pack_indices<AI...>, pack_indices<CI...>, CallArgs&&... args)
		{
			return this->callFunc(get<AI>(args...).hostProxy()..., get<CI>(args...)...);
		}
		
	public:
		
		template<typename... CallArgs>
		Ret operator()(CallArgs&&... args)
		{
			static_assert(sizeof...(CallArgs) == numArgs, "Number of arguments not matching Call function");
			return apply(any_indices, const_indices, std::forward<CallArgs>(args)...);
		}
	
	private:
		
		CallImpl(CallFunc call): callFunc(call) {}
		CallFunc callFunc;
		
		friend CallImpl<Ret, Args...> CallWrapper<Ret, Args...>(CallFunc);
	};
	
} // end namespace skepu2
