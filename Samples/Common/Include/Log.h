#pragma once

#ifdef DDEBUG
// #define LOG_DEBUG(...) std::printf(__VA_ARGS__);
#define LOG_DEBUG(...) \
	if (true) { \
		std::printf(__VA_ARGS__); \
		std::printf("\n"); \
	}
#else
#define LOG_DEBUG(...)
#endif
