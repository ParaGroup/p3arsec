#!/bin/sh

echo "Remove all PARSEC files and restore the repository..."
rm -rf bin config/ ext/ log/ man/ pkgs/ toolkit/ .parsec_uniquefile CHANGELOG CONTRIBUTORS env.sh FAQ LICENSE_PARSEC README_PARSEC repairpod.sh version
echo "Done."