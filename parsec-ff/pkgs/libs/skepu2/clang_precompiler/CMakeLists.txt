set(LLVM_LINK_COMPONENTS support)
set(LLVM_USED_LIBS clangTooling clangBasic clangAST)

add_clang_executable(skepu-tool
  skepu.cpp
  visitor.cpp
  data_structures.cpp
  code_gen.cpp
  mapreduce_cl.cpp
  mapreduce_cu.cpp
  map_cl.cpp
  map_cu.cpp
  scan_cl.cpp
  scan_cu.cpp
  reduce_cl.cpp
  reduce_cu.cpp
  mapoverlap_cl.cpp
  mapoverlap_cu.cpp
  call_cl.cpp
  call_cu.cpp
  )
target_link_libraries(skepu-tool
  clangTooling
  clangBasic
  clangASTMatchers
  )
