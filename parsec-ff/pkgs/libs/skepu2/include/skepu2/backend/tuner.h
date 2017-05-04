#pragma once

#include <skepu2/backend/benchmark.h>

#define MEASURE_REPEATS 9
#define ARG_SIZE_STEP_FACTOR 4
#define ARG_SIZE_MIN 16
#define ARG_SIZE_MAX (1 << 18)

namespace skepu2
{	
	namespace backend
	{
		namespace tuner
		{
			template<typename T>
			struct container_args_tuple_transformer
			{};
			
			// Requires each of ´Types´ either be ´Vec<T>´ or ´Matrix<T>´ for some type ´T´
			template<typename... Types>
			struct container_args_tuple_transformer<std::tuple<Types...>>
			{
				using type = std::tuple<typename std::decay<Types>::type::ContainerType...>;
			};
			
			template<typename C>
			void container_update_host_invalidate(C &c)
			{
				c.updateHostAndInvalidateDevice();
			}
			
			template<class Tuple, size_t...Is>
			void update_host_invalidate_all_in_tuple(Tuple&& tuple, future_std::index_sequence<Is...>)
			{
				int hej[] = { 0, (container_update_host_invalidate(std::get<Is>(std::forward<Tuple>(tuple))), 0)... };
			}
			
			template<class Tuple>
			void update_host_invalidate_all_in_tuple(Tuple&& tuple)
			{
				update_host_invalidate_all_in_tuple(std::forward<Tuple>(tuple),
					typename future_std::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>::type());
			}
			
			template<typename T>
			void container_reserve(skepu2::Vector<T> &v, size_t size)
			{
				v.reserve(size);
			}
			
			
			template<typename T>
			void container_reserve(skepu2::Matrix<T> &m, size_t size)
			{
			//	c.reserve(size);
			}
			
			template<typename T, typename Skeleton>
			void container_resize(skepu2::Vector<T> &v, size_t size, Skeleton &, bool = false)
			{
				v.resize(size);
			}
			
			template<typename T, typename Skeleton>
			typename std::enable_if<Skeleton::skeletonType != SkeletonType::MapOverlap2D, void>::type
			container_resize(skepu2::Matrix<T> &m, size_t size, Skeleton &, bool isResult = false)
			{
			//	size_t side = (size_t)sqrt((double)size);
			//	m.resize(side, side);
				m.resize(size, size);
			}
			
			template<typename T, typename Skeleton>
			typename std::enable_if<Skeleton::skeletonType == SkeletonType::MapOverlap2D, void>::type
			container_resize(skepu2::Matrix<T> &m, size_t size, Skeleton &s, bool isResult = false)
			{
				size_t side = (size_t)sqrt((double)size);
				if (isResult)
				{
					size_t height = side - 2 * s.getOverlap().first;
					size_t width  = side - 2 * s.getOverlap().second; 
					m.resize(height, width);
					std::cout << "Resizing to " << height << " x " << width << "\n";
				}
				else
				{
					m.resize(side, side);
					std::cout << "Resizing to " << side << " x " << side << "\n";
				}
			}
			
			template<class Tuple, size_t...Is>
			void reserve_all_in_tuple(Tuple&& tuple, size_t size, future_std::index_sequence<Is...>)
			{
				int hej[] = { 0, (container_reserve(std::get<Is>(std::forward<Tuple>(tuple)), size), 0)... };
			}
			
			template<class Tuple>
			void reserve_all_in_tuple(Tuple&& tuple, size_t size)
			{
				reserve_all_in_tuple(std::forward<Tuple>(tuple), size,
					typename future_std::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>::type());
			}
			
			template<class Tuple, typename Skeleton, size_t...Is>
			void resize_all_in_tuple(Tuple&& tuple, size_t size, Skeleton &s, bool isResult, future_std::index_sequence<Is...>)
			{
				int hej[] = { 0, (container_resize(std::get<Is>(std::forward<Tuple>(tuple)), size, s, isResult), 0)... };
			}
			
			template<class Tuple, typename Skeleton>
			void resize_all_in_tuple(Tuple&& tuple, size_t size, Skeleton &s, bool isResult = false)
			{
				resize_all_in_tuple(std::forward<Tuple>(tuple), size, s, isResult,
					typename future_std::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>::type());
			}
			
			
#ifdef SKEPU_PRECOMPILED
			
			template<typename Skeleton, size_t... ResultIdx, size_t... ElwiseIdx, size_t... ContainerIdx, size_t... UniformIdx>
			void tune_impl(Skeleton& instance, const size_t min, const size_t max, const size_t factor, const size_t repeats,
				future_std::index_sequence<ResultIdx...>,    future_std::index_sequence<ElwiseIdx...>,
				future_std::index_sequence<ContainerIdx...>, future_std::index_sequence<UniformIdx...>)
			{
				// Tuple with container for the output container (can be empty)
				typename select_if<Skeleton::prefers_matrix,
					typename add_container_layer<Matrix, typename Skeleton::ResultArg>::type,
					typename add_container_layer<Vector, typename Skeleton::ResultArg>::type>::type resultArg;
				
				// Tuple with containers for the element-wise arguments (can be empty)
				typename select_if<Skeleton::prefers_matrix,
					typename add_container_layer<Matrix, typename Skeleton::ElwiseArgs>::type,
					typename add_container_layer<Vector, typename Skeleton::ElwiseArgs>::type>::type elwiseArgs;
				
				// Tuple with containers for the random-access container arguments (can be different container types, can be empty)
				typename container_args_tuple_transformer<typename Skeleton::ContainerArgs>::type containerArgs;
				
				// Tuple with uniform arguments (can be empty)
				typename Skeleton::UniformArgs uniformArgs;
				
				// Reserve memory for the maximum input size in advance
				reserve_all_in_tuple(resultArg,  max);
				reserve_all_in_tuple(elwiseArgs, max);
				
				// The plan which will be generated
				ExecPlan *plan = new ExecPlan();
				plan->setCalibrated();
				
				// Run tests for all input sizes
				for (size_t i = min, prev_i = 0; i <= max; prev_i = i, i *= factor)
				{
					resize_all_in_tuple(resultArg,  i, instance, true);
					resize_all_in_tuple(elwiseArgs, i, instance);
					resize_all_in_tuple(containerArgs, i, instance);
					
					auto mintime = benchmark::TimeSpan::max();
					BackendSpec bestBackendSpec;
					
					// Run tests for all available backends
					for (auto backend : Backend::availableTypes())
					{
						BackendSpec spec{backend};
						instance.setBackend(spec);
						
						auto duration = benchmark::basicBenchmark(
							MEASURE_REPEATS, i,
							[&] (size_t) {
								instance(
									std::get<ResultIdx>(resultArg)...,
									std::get<ElwiseIdx>(elwiseArgs)...,
									std::get<ContainerIdx>(containerArgs)...,
									std::get<UniformIdx>(uniformArgs)...);
							},
							[&](benchmark::TimeSpan duration)
							{
								skepu2::containerutils::updateHostAndInvalidateDevice(
									std::get<ResultIdx>(resultArg)...,
									std::get<ElwiseIdx>(elwiseArgs)...,
									std::get<ContainerIdx>(containerArgs)...);
							}
						);
						
						// If best time, save this
						if (duration < mintime)
						{
							mintime = duration;
							bestBackendSpec = spec;
						}
					}
					
					plan->add(prev_i, i, bestBackendSpec);
				}
				
				instance.setExecPlan(std::move(plan));
			}
			
			
			template<typename Skeleton>
			void tune(Skeleton& instance, size_t maxSize = ARG_SIZE_MAX)
			{
				tune_impl(instance, ARG_SIZE_MIN, maxSize, ARG_SIZE_STEP_FACTOR, MEASURE_REPEATS,
					typename future_std::make_index_sequence<std::tuple_size<typename Skeleton::ResultArg>::value>::type(),
					typename future_std::make_index_sequence<std::tuple_size<typename Skeleton::ElwiseArgs>::value>::type(),
					typename future_std::make_index_sequence<std::tuple_size<typename Skeleton::ContainerArgs>::value>::type(),
					typename future_std::make_index_sequence<std::tuple_size<typename Skeleton::UniformArgs>::value>::type());
			}
			
#else
			
			template<typename S>
			void tune(S& instance)
			{
				// Do nothing
			}
			
#endif
		}
	}
}
