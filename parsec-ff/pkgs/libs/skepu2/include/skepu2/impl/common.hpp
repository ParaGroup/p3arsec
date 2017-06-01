#pragma once

#define REQUIRES(...) typename std::enable_if<(__VA_ARGS__), bool>::type = 0
#define REQUIRES_DEF(...) typename std::enable_if<(__VA_ARGS__), bool>::type

#define MAX_SIZE ((size_t)-1)

#define VARIANT_OPENCL(block)
#define VARIANT_CUDA(block)
#define VARIANT_OPENMP(block)
#define VARIANT_CPU(block) block

#include <vector>
#include <iostream>
#include <utility>
#include <cassert>
#include <algorithm>
#include <functional>

namespace skepu2
{
	// ----------------------------------------------------------------
	// sizes and indices structures
	// ----------------------------------------------------------------
	
	struct Index1D
	{
		size_t i;
	};

	struct Index2D
	{
		size_t row;
		size_t col;
	};
	
	
	enum class AccessMode
	{
		Read,
		Write,
		ReadWrite,
		None
	};
	
	static inline constexpr bool hasReadAccess(AccessMode m)
	{
		return m == AccessMode::Read || m == AccessMode::ReadWrite;
	}
	
	static inline constexpr bool hasWriteAccess(AccessMode m)
	{
		return m == AccessMode::Write || m == AccessMode::ReadWrite;
	}
	
	enum class SkeletonType
	{
		Map,
		MapReduce,
		Reduce1D,
		Reduce2D,
		Scan,
		MapOverlap1D,
		MapOverlap2D,
		Call,
	};
	
	
#ifdef SKEPU_OPENCL
	
	/*!
	 * helper to return data type in a string format using template specialication technique.
	 * Compile-time error if no overload is found.
	 */
	template<typename T>
	inline std::string getDataTypeCL();
	
	template<> inline std::string getDataTypeCL<char>          () { return "char";           }
	template<> inline std::string getDataTypeCL<unsigned char> () { return "unsigned char";  }
	template<> inline std::string getDataTypeCL<short>         () { return "short";          }
	template<> inline std::string getDataTypeCL<unsigned short>() { return "unsigned short"; }
	template<> inline std::string getDataTypeCL<int>           () { return "int";            }
	template<> inline std::string getDataTypeCL<unsigned int>  () { return "unsigned int";   }
	template<> inline std::string getDataTypeCL<long>          () { return "long";           }
	template<> inline std::string getDataTypeCL<unsigned long> () { return "unsigned long";  }
	template<> inline std::string getDataTypeCL<float>         () { return "float";          }
	template<> inline std::string getDataTypeCL<double>        () { return "double";         }
	
#endif
	
	// Dummy base class for sequential skeleton classes.
	// Includes empty member functions which has no meaning in a sequential context.
	class SeqSkeletonBase
	{
	public:
		void setBackend(BackendSpec) {}
		void resetBackend() {}
		void setExecPlan(ExecPlan *plan)
		{
			delete plan;
		}
		
		template<typename... Args>
		void tune(Args&&... args) { }
	};
}

#include "meta_helpers.hpp"
#include "skepu2/vector.hpp"
#include "skepu2/matrix.hpp"
#include "skepu2/sparse_matrix.hpp"

namespace skepu2
{
	inline size_t elwise_width(std::tuple<>)
	{
		return 0;
	}
	
	template<typename... Args>
	inline size_t elwise_width(std::tuple<Args...> &t)
	{
		return std::get<0>(t).getParent().total_cols();
	}
	
	// ----------------------------------------------------------------
	// is_skepu_{vector|matrix|container} trait classes
	// ----------------------------------------------------------------
	
	template<typename T>
	struct is_skepu_matrix: std::false_type {};
	
	template<typename T>
	struct is_skepu_matrix<skepu2::Matrix<T>>: std::true_type {};
	
	template<typename T>
	struct is_skepu_matrix<skepu2::SparseMatrix<T>>: std::true_type {};
	
	template<typename T>
	struct is_skepu_vector: std::false_type {};
	
	template<typename T>
	struct is_skepu_vector<skepu2::Vector<T>>: std::true_type {};
	
	template<typename T>
	struct is_skepu_container:
		std::integral_constant<bool,
			is_skepu_vector<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value ||
			is_skepu_matrix<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value> {};
		
		
		
	template<typename T>
	struct is_skepu_vector_proxy: std::false_type {};
	
	template<typename T>
	struct is_skepu_vector_proxy<skepu2::Vec<T>>: std::true_type {};
	
	template<typename T>
	struct is_skepu_matrix_proxy: std::false_type {};
	
	template<typename T>
	struct is_skepu_matrix_proxy<skepu2::Mat<T>>: std::true_type {};
	
	template<typename T>
	struct is_skepu_matrix_proxy<skepu2::SparseMat<T>>: std::true_type {};
	
	template<typename T>
	struct is_skepu_container_proxy:
		std::integral_constant<bool,
			is_skepu_vector_proxy<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value ||
			is_skepu_matrix_proxy<typename std::remove_cv<typename std::remove_reference<T>::type>::type>::value> {};
	
	
	
	// ----------------------------------------------------------------
	// is_skepu_iterator trait class
	// ----------------------------------------------------------------
	
	template<typename T, typename Ret>
	struct is_skepu_iterator: bool_constant<
		std::is_same<T, typename Vector<Ret>::iterator>::value ||
		std::is_same<T, typename Vector<Ret>::const_iterator>::value ||
		std::is_same<T, typename Matrix<Ret>::iterator>::value
	> {};
	
	
	// ----------------------------------------------------------------
	// is_skepu_index trait class
	// ----------------------------------------------------------------
	
	template<typename T>
	struct is_skepu_index: std::false_type{};
	
	template<>
	struct is_skepu_index<Index1D>: std::true_type{};
	
	template<>
	struct is_skepu_index<Index2D>: std::true_type{};
	
	
	template<typename... Args>
	struct is_indexed
	: bool_constant<is_skepu_index<typename pack_element<0, Args...>::type>::value> {};
	
	template<>
	struct is_indexed<>
	: std::false_type{};
	
	
	// ----------------------------------------------------------------
	// ConditionalIndexForwarder utility structure
	// ----------------------------------------------------------------
	
	template<bool indexed, typename Func>
	struct ConditionalIndexForwarder
	{
		using Ret = typename return_type<Func>::type;
		
		template<typename... CallArgs>
		static Ret forward(Func func, Index1D i, CallArgs&&... args)
		{
			return func(i, std::forward<CallArgs>(args)...);
		}
		
		template<typename... CallArgs>
		static Ret forward(Func func, Index2D i, CallArgs&&... args)
		{
			return func(i, std::forward<CallArgs>(args)...);
		}
		
		
		template<typename... CallArgs>
		static Ret forward_device(Func func, Index1D i, CallArgs&&... args)
		{
			return func(i, std::forward<CallArgs>(args)...);
		}
		
		template<typename... CallArgs>
		static Ret forward_device(Func func, Index2D i, CallArgs&&... args)
		{
			return func(i, std::forward<CallArgs>(args)...);
		}
	};
	
	template<typename Func>
	struct ConditionalIndexForwarder<false, Func>
	{
		using Ret = typename return_type<Func>::type;
		
		template<typename... CallArgs>
		static Ret forward(Func func, Index1D, CallArgs&&... args)
		{
			return func(std::forward<CallArgs>(args)...);
		}
		
		template<typename... CallArgs>
		static Ret forward(Func func, Index2D, CallArgs&&... args)
		{
			return func(std::forward<CallArgs>(args)...);
		}
		
		
		template<typename... CallArgs>
		static Ret forward_device(Func func, Index1D, CallArgs&&... args)
		{
			return func(std::forward<CallArgs>(args)...);
		}
		
		template<typename... CallArgs>
		static Ret forward_device(Func func, Index2D, CallArgs&&... args)
		{
			return func(std::forward<CallArgs>(args)...);
		}
	};
	
}
