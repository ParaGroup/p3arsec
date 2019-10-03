#!/bin/sh

rsync -av parsec-ff/* ./
rsync -av parsec-hooks/* ./


# rm -r pkgs/apps/blackschols/inst/amd64-linux.gcc-caf
# rm -r pkgs/apps/blackschols/obj/amd64-linux.gcc-caf
# ./bin/parsecmgmt -a build -p blackscholes -c gcc-caf
# rm -r pkgs/apps/blackscholes/inst/amd64-linux.gcc-ff
# rm -r pkgs/apps/blackschols/obj/amd64-linux.gcc-ff
# ./bin/parsecmgmt -a build -p blackscholes -c gcc-ff

# rm -r pkgs/apps/ferret/inst/amd64-linux.gcc-caf
# rm -r pkgs/apps/ferret/obj/amd64-linux.gcc-caf
# ./bin/parsecmgmt -a build -p ferret -c gcc-caf
# rm -r pkgs/apps/ferret/inst/amd64-linux.gcc-ff
# rm -r pkgs/apps/ferret/obj/amd64-linux.gcc-ff
# ./bin/parsecmgmt -a build -p ferret -c gcc-ff

# rm -r pkgs/apps/raytrace/inst/amd64-linux.gcc-caf
# rm -r pkgs/apps/raytrace/obj/amd64-linux.gcc-caf
# ./bin/parsecmgmt -a build -p raytrace -c gcc-caf
# rm -r pkgs/apps/raytrace/inst/amd64-linux.gcc-ff
# rm -r pkgs/apps/raytrace/obj/amd64-linux.gcc-ff
# ./bin/parsecmgmt -a build -p raytrace -c gcc-ff


# rm -r pkgs/kernels/canneal/inst/amd64-linux.gcc-caf
# rm -r pkgs/kernels/canneal/obj/amd64-linux.gcc-caf
# ./bin/parsecmgmt -a build -p canneal -c gcc-caf
# rm -r pkgs/kernels/canneal/inst/amd64-linux.gcc-ff
# rm -r pkgs/kernels/canneal/obj/amd64-linux.gcc-ff
# ./bin/parsecmgmt -a build -p canneal -c gcc-ff
