#pragma once

#ifndef _ERR_STD
#define _ERR_STD ::std::
#endif

#define _CHECK_VULKAN_RESULT2(func, line) \
	{ \
		::vk::Result local_result_##line = ::vk::Result(func); \
		if (local_result_##line != ::vk::Result::eSuccess) [[unlikely]] { \
			_ERR_STD printf("Vulkan error: %s " #func " in " __FILE__ ":" #line, ::vk::to_string(local_result_##line).c_str()); \
			_ERR_STD exit(1); \
		} \
	}

#define _CHECK_VULKAN_RESULT(func, line) _CHECK_VULKAN_RESULT2(func, line)
#define CHECK_VULKAN_RESULT(func) _CHECK_VULKAN_RESULT(func, __LINE__)

#define CHECK(expr) \
	{ \
		auto const ret = (expr); \
		if (ret != decltype(ret){}) { \
			_ERR_STD printf("Error: %s\n", #expr); \
			_ERR_STD exit(1); \
		} \
	}
