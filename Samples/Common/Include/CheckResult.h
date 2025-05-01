#pragma once

#define _CHECK_VULKAN_RESULT2(func, line) \
	{ \
		::vk::Result local_result_##line = ::vk::Result(func); \
		if (local_result_##line != ::vk::Result::eSuccess) [[unlikely]] { \
			::std::printf("Vulkan error: %s " #func " in " __FILE__ ":" #line, ::vk::to_string(local_result_##line).c_str()); \
			::std::exit(1); \
		} \
	}

#define _CHECK_VULKAN_RESULT(func, line) _CHECK_VULKAN_RESULT2(func, line)
#define CHECK_VULKAN_RESULT(func) _CHECK_VULKAN_RESULT(func, __LINE__)
