#pragma once

#define CHECK_RESULT(func) \
	{ \
		vk::Result local_result_ = (func); \
		if (local_result_ != vk::Result::eSuccess) { \
			std::printf("Error: %s in %s %s:%d\n", vk::to_string(local_result_).c_str(), #func, __FILE__, __LINE__); \
			std::exit(1); \
		} \
	}