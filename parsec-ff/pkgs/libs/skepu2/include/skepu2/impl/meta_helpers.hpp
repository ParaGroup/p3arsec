#pragma once

#include <type_traits>

template<bool B>
struct bool_constant : std::integral_constant<bool, B> { };

template<size_t I>
struct index_constant : std::integral_constant<size_t, I> { };

/////////////////////////////
// true_for_n
//
// Higher order variadic template meta-function.
// Checks if a meta-predicate is true for the N first parameters.

template<template<typename> class Pred, int N, typename... Types>
struct true_for_n;

template<template<typename> class Pred, int N>
struct true_for_n<Pred, N>: bool_constant<N == 0> {};

template<template<typename> class Pred, int N, typename First>
struct true_for_n<Pred, N, First> :
bool_constant<N == 0 || (Pred<First>::value && N == 1)> {};

template<template<typename> class Pred, int N, typename First, typename... Rest>
struct true_for_n<Pred, N, First, Rest...> :
bool_constant<N == 0 || (Pred<First>::value && (N == 1 || true_for_n<Pred, N-1, Rest...>::value))> {};


/////////////////////////////
// Trait count
template<template<typename> class Pred, typename... Types>
struct trait_count_first: index_constant<0> {};

template<template<typename> class Pred, typename First, typename... Rest>
struct trait_count_first<Pred, First, Rest...> :
index_constant<Pred<First>::value ? 1 + trait_count_first<Pred, Rest...>::value : 0> {};

template<template<typename> class Pred, typename...>
struct trait_count_all {};

template<template<typename> class Pred, typename First, typename... Rest>
struct trait_count_all<Pred, First, Rest...> :
index_constant<(Pred<First>::value ? 1 : 0) + trait_count_all<Pred, Rest...>::value> {};

template<template<typename> class Pred>
struct trait_count_all<Pred> :
index_constant<0> {};

template<template<typename> class Pred, typename... Types>
struct trait_count_first_not: index_constant<0> {};

template<template<typename> class Pred, typename First, typename... Rest>
struct trait_count_first_not<Pred, First, Rest...> :
index_constant<!Pred<First>::value ? 1 + trait_count_first_not<Pred, Rest...>::value : 0> {};


/////////////////////////////
// pack_element
template<size_t I, typename First, typename... Rest>
struct pack_element
{
	static_assert(I < 1+sizeof...(Rest), "Type index is out of bounds!");
	using type = typename pack_element<I-1, Rest...>::type;
};

template<typename First, typename... Rest>
struct pack_element<0, First, Rest...>
{
	using type = First;
};

/////////////////////////////
// get
template <typename R, size_t Ip, size_t Ij, typename... Tp>
struct get_impl
{
	static R& dispatch(Tp...);
};

template<class R,  size_t Ip, size_t Jp, typename Head, typename... Tp>
struct get_impl<R, Ip, Jp, Head, Tp...>
{
	static R& dispatch(Head&, Tp&... tps)
	{
		return get_impl<R, Ip, Jp + 1, Tp...>::dispatch(tps...);
	}
};

template<size_t Ip, typename Head, typename... Tp>
struct get_impl<Head, Ip, Ip, Head, Tp...>
{
	static Head& dispatch(Head& h, Tp&...)
	{
		return h;
	}
};

/*template <size_t Ip, typename... Tp>
typename pack_element<Ip, Tp...>::type&
get(Tp&&... tps)
{
	return get_impl<typename pack_element<Ip, Tp...>::type, Ip, 0, Tp...>::dispatch(tps...);
}*/

template <size_t Ip, typename... Tp>
inline typename pack_element<Ip, Tp...>::type&
get(Tp&... tps)
{
	return get_impl<typename pack_element<Ip, Tp...>::type, Ip, 0, Tp...>::dispatch(tps...);
}

template <typename... Tp>
inline auto get_any(Tp&&... tps) -> decltype(get<0>(tps...))
{
	return get<0>(tps...);
}

/////////////////////////////
// pack indices (basically C++14 std::index_sequence)

template <size_t...>
struct pack_indices {};

template <size_t Sp, class IntPack, size_t Ep>
struct make_indices_imp;

template <size_t Sp, size_t ... Indices, size_t Ep>
struct make_indices_imp<Sp, pack_indices<Indices...>, Ep>
{
	typedef typename make_indices_imp<Sp+1, pack_indices<Indices..., Sp>, Ep>::type type;
};

template <size_t Ep, size_t ... Indices>
struct make_indices_imp<Ep, pack_indices<Indices...>, Ep>
{
	typedef pack_indices<Indices...> type;
};

template <size_t Ep, size_t Sp = 0>
struct make_pack_indices
{
	static_assert(Sp <= Ep, "__make_tuple_indices input error");
	typedef typename make_indices_imp<Sp, pack_indices<>, Ep>::type type;
};


namespace future_std
{
	template <size_t... S>
	using index_sequence = pack_indices<S...>;
	
	template <size_t Ep>
	using make_index_sequence = make_pack_indices<Ep>;
}


/////////////////////////////
// is_all_same

template<template<typename,typename>class checker, typename... Ts>
struct is_all : std::true_type {};

template<template<typename,typename>class checker, typename T0, typename T1, typename... Ts>
struct is_all<checker, T0, T1, Ts...> :
  bool_constant<checker<T0, T1>::value && is_all<checker, T0, Ts...>::value> {};

template<typename... Ts>
using is_all_same = is_all< std::is_same, Ts... >;

/////////////////////////////
// filter

template<template<class> class, template <class...> class, class...>
struct filter;

template<template<class> class Pred, template <class...> class Variadic>
struct filter<Pred, Variadic>
{
	using type = Variadic<>;
};

template<template<class> class Pred, template <class...> class Variadic, class T, class... Ts>
struct filter<Pred, Variadic, T, Ts...>
{
	template<class, class> struct Cons;
	
	template<class Head, class... Tail>
	struct Cons<Head, Variadic<Tail...> >
	{
		using type = Variadic<Head, Tail...>;
	};
		
	using rest = typename filter<Pred, Variadic, Ts...>::type;
	using type = typename std::conditional<Pred<T>::value, typename Cons<T, rest>::type, rest>::type;
};

// return type
template <class F>
struct return_type;

template <class R, class... A>
struct return_type<R (*)(A...)> { typedef R type; };

template <class R, class... A>
struct return_type<std::function<R(A...)>> { typedef R type; };

// parameter type
template <size_t I, class F>
struct parameter_type;

template <size_t I, class Ret, class... Params>
struct parameter_type<I, Ret (*)(Params...)>
{
	using type = typename pack_element<I, Params...>::type;
};



template <class F>
struct func_arity;

template <class R, class... A>
struct func_arity<R (*)(A...)>: index_constant<sizeof...(A)> {};


/////////////////////////////
// is_arg_ptr_to_constant
//
// Find out whether the I:th argument of a userfunction is a pointer-to-constant.
// Example use: Deducing the access mode of "any" containers in Map and MapReduce.

template<size_t I, typename First, typename... Rest>
struct is_arg_ptr_to_constant_impl
: bool_constant<is_arg_ptr_to_constant_impl<I - 1, Rest...>::value> { static_assert(I != 0, "Error"); };

template<typename First, typename... Rest>
struct is_arg_ptr_to_constant_impl<0, First, Rest...>
: bool_constant<std::is_const<typename std::remove_pointer<First>::type>::value> { };

template<size_t I, typename UserFunctionStruct, typename Ret, typename... Args>
constexpr bool is_arg_ptr_to_constant_f(Ret(*func)(Args...))
{
	return is_arg_ptr_to_constant_impl<I + (UserFunctionStruct::indexed ? 1 : 0), Args...>::value;
}

template<size_t I, typename MapFunc>
struct is_arg_ptr_to_constant
: bool_constant<is_arg_ptr_to_constant_f<I, MapFunc>(MapFunc::CPU)> {};


/////////////////////////////
// all_equal
//
// Checks that all parameters to a function call are equal to the first parameter.
// Per the == operator.
// Used e.g. in skeleton invocation to verify that all container arguments have a correct size.

template<typename T>
inline bool all_equal(T)
{
	return true;
}

template<typename T, typename First, typename... Rest>
inline bool all_equal(T&& ref, First&& first, Rest&&... rest)
{
	return ref == first && all_equal(std::forward<T>(ref), std::forward<Rest>(rest)...);
}


/////////////////////////////
// pack_expand
//
// A 'hack' to expand expressions in contexts where this is not normally possible.
// Note that evaluation order of function calls is undefined!
// An optimizing compiler should remove the function call since the body is empty.

template< typename... Parms>
inline void pack_expand(Parms...)
{}


/////////////////////////////
// variadic conjunction & disjunction
//
// While waiting for C++17 fold expressions...

inline bool conjunction()
{
	return true;
}

template<typename First, typename... Rest>
inline bool conjunction(First&& first, Rest&&... rest)
{
	return first && conjunction(rest...);
}

inline bool disjunction()
{
	return false;
}

template<typename First, typename... Rest>
inline bool disjunction(First&& first, Rest&&... rest)
{
	return first || disjunction(rest...);
}


////////////////////////////////
// Lambda support

template<typename T>
struct memfun_type {};

template<typename Ret, typename Class, typename... Args>
struct memfun_type<Ret(Class::*)(Args...) const>
{
    using type = std::function<Ret(Args...)>;
};

template<typename Ret, typename Class, typename... Args>
struct memfun_type<Ret(Class::*)(Args...)>
{
    using type = std::function<Ret(Args...)>;
};

// Cast a callable object to std::function (const-qualified)
template<typename F>
typename memfun_type<decltype(&F::operator())>::type
lambda_cast(F const &func)
{
    return func;
}

// Cast a const callable object to std::function
template<typename F>
typename memfun_type<decltype(&F::operator())>::type
lambda_cast(F &func)
{
    return func;
}


/////////////////////////////
// Transform a tuple of types to a tuple of some ´Container´:s of the same types

template<template<class> class Container, typename T>
struct add_container_layer
{};

template<template<class> class Container, typename... Types>
struct add_container_layer<Container, std::tuple<Types...>>
{
	using type = std::tuple<Container<typename std::decay<Types>::type>...>;
};


// Apply some functor ´Func´ with the first argument taken from a tuple, for each element in the tuple,
// and some number of other arguments

template<class Func, class Tuple, size_t...Is, typename... Args>
inline void for_each_in_tuple(Func f, Tuple&& tuple, Args&&... args, future_std::index_sequence<Is...>)
{
	int dummy[] = { 0, ((void)f(std::get<Is>(std::forward<Tuple>(tuple)), std::forward<Args>(args)...), 0)... };
}

template<class Func, class Tuple, typename... Args>
inline void for_each_in_tuple(Func f, Tuple&& tuple, Args&&... args)
{
	for_each_in_tuple(f, std::forward<Tuple>(tuple), std::forward<Args>(args)...,
		typename future_std::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>::type());
}

// Select between two possible types based on some boolean ´Condition´

template<bool Condition, typename T, typename F>
struct select_if;

template<typename T, typename F>
struct select_if<true, T, F> { using type = T; };

template<typename T, typename F>
struct select_if<false, T, F> { using type = F; };



