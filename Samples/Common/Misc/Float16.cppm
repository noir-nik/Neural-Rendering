export module float16;
import std;

export extern "C++" {
#define FLOAT_16_MODULE
#include "Float16.hpp"
}
