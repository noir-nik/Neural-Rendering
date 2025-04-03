# Neural rendering

## Requirements

- Vukan SDK >= 1.4.309
- Compiler with C++23 support
- Toolchain with C++23 import std
- CMake with Ninja build system

## Installation

1. Clone the Repository:

```sh
git clone https://github.com/noir-nik/Neural-Rendering.git --recurse-submodules
```

2. Build with ninja, clang and libc++:

```sh
cd Neural-Rendering
cmake . -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_FLAGS="-stdlib=libc++"
cmake --build build
```

## Run Sdf sample from the repository root:

```sh
./build/Samples/SDF/SDF
```
