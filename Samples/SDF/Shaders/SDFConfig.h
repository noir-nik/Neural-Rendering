#ifndef SDFCONFIG_H
#define SDFCONFIG_H

#ifndef COMPONENT_BITS
#define COMPONENT_BITS 16
#endif

// #ifndef COOPVEC_TYPE
// #define COOPVEC_TYPE float16_t
// #endif

#ifndef _COMPONENT_TYPE
#define _COMPONENT_TYPE Float16
#endif

// #ifndef _EVAL
// #define _EVAL(...) __VA_ARGS__
// #endif

#ifdef __cplusplus
#define _CONCAT(x, y) x##y
#define _FWD(macro, ...) macro(__VA_ARGS__)
#define COMPONENT_TYPE _FWD(_CONCAT, vk::ComponentTypeKHR::e, _COMPONENT_TYPE)
#else
#define COMPONENT_TYPE CoopVecComponentType::_COMPONENT_TYPE
#endif // __cplusplus

#endif // SDFCONFIG_H
