# SkePU 2

SkePU 2 consists of four parts:

1. A sequential interface and accompanying implementation for C++ skeleton programming.
2. A source-to-source precompiler tool built on top of the Clang C-language compiler front-end,
	transforming code written for the sequential interface for parallel and heterogeneous execution.
3. Minor patches to Clang to support said tool.
4. A collection of parallel and/or heterogeneous back-ends targeting various architectures and systems.

## Setting up the SkePU source-to-source compiler

1. Clone LLVM and Clang Git repositories and confirm that Clang builds.
2. Patch Clang (see below) to add SkePU attributes and diagnostics to Clang.
	It also instructs the Clang build system to build the SkePU tool.
3. Create a directory symlink in `<clang source>/tools` named `skepu` pointing to `clang_precompiler`.
	Clang will now find the source files for the SkePU tool.
4. Run `ninja skepu` in your Clang build directory.
5. If successful, the SkePU precompiler binary should now be in `<clang build>/bin/skepu`.
6. Run `skepu -help` to confirm that everything worked and to see available options.

### Cloning LLVM and Clang

`$ mkdir ~/clang-llvm && cd ~/clang-llvm`

`$ git clone http://llvm.org/git/llvm.git`

`$ cd llvm/tools`

`$ git clone http://llvm.org/git/clang.git`


### Patch Clang

TODO: Checkout a specific llvm / Clang version first!

`$ cd <clang source>`

`$ git apply <skepu source>/clang_patch.patch`

### Symlink SkePU sources in Clang

`$ ln -s <skepu source>/clang_precompiler <clang source>/tools/skepu-tool`

### Building SkePU precompiler

Create and/or move to build directory.

`$ cd <clang build>`

Set up CMake (once)

`$ cmake -G "Unix Makefiles" <llvm source> -DCMAKE_BUILD_TYPE=Release`

(Release mode is faster and much more space efficient, remove option to build in Debug mode.)

Build the SkePU tool (once for SkePU users)

`$ make skepu-tool`


## Compatibility with SkePU 1

SkePU 1 code is not compatible with SkePU 2 and vice-versa. SkePU 2 is in large part based on concepts from SkePU 1, and the data structures are the same, so it should be fairly straightworward to port a SkePU 1 project to SkePU 2. It may require some effort to fit the SkePU 2 precompiler into a large project with non-trivial build system, however.

## Directory layout


### `include`

As SkePU 2 is a header library, this directory contains headers and source files for the SkePU runtime.

#### `include/skepu2`

Contains the serial skeleton interface (headers with inline implementations), along with headers for SkePU containers.

##### `include/skepu2/impl`

Helpers.

##### `include/skepu2/backend`

Contains header files for the SkePU skeleton backends.

##### `include/skepu2/backend/impl`

Contains implementations of the various SkePU skeleton backends and containers.

### `examples`

Contains SkePU 2 example programs.

### `clang_precompiler`

Sources for the source-to-source Clang tool. This directory is part of the LLVM directory tree, and a symlink named `<path_to_llvm>/llvm/tools/clang/tools/skepu` should link here.
