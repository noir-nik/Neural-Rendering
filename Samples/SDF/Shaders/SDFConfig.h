#ifndef SDFCONFIG_H
#define SDFCONFIG_H

#define COOPVEC_TYPE float16_t

#define SRC_COMPONENT_TYPE vk::ComponentTypeKHR::eFloat16
#define DST_MATRIX_TYPE vk::ComponentTypeKHR::eFloat16
#define DST_VECTOR_TYPE vk::ComponentTypeKHR::eFloat16
// #define MATRIX_LAYOUT vk::CooperativeVectorMatrixLayoutNV::eRowMajor
#define MATRIX_LAYOUT vk::CooperativeVectorMatrixLayoutNV::eInferencingOptimal

#define COMPONENT_TYPE CoopVecComponentType::Float16

#endif // SDFCONFIG_H
