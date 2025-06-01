module;
#include <stdio.h>
#include <stdlib.h>

export module StbModule;

#ifdef EXPORT
#undef EXPORT
#endif

#define EXPORT export extern "C"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winclude-angled-in-module-purview"
// #pragma clang diagnostic ignored "-Wmacro-redefined"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 5244)
#endif

EXPORT {
#include "stb_image.h"
#include "stb_image_write.h"
#include "stb_rect_pack.h"
#include "stb_truetype.h"
}

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
