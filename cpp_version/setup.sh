#!/bin/bash

# Create directory structure
mkdir -p cpp_version/src cpp_version/include cpp_version/models cpp_version/build cpp_version/bin cpp_version/lib cpp_version/third_party

# Download dr_wav
curl -L https://raw.githubusercontent.com/mackron/dr_libs/master/dr_wav.h -o cpp_version/third_party/dr_wav.h 