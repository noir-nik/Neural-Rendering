#pragma once

#ifdef NO_LOGGING
#define LOG_TRACE(fmt, ...)
#define LOG_INFO(fmt, ...)
#define LOG_WARN(fmt, ...)
#define LOG_ERROR(fmt, ...)
#else // NO_LOGGING
#define LOG_TRACE(fmt, ...) \
	Logger::Get().LogFormat(LogLevel::Trace, fmt, ## __VA_ARGS__)

#define LOG_INFO(fmt, ...) \
	Logger::Get().LogFormat(LogLevel::Info, fmt, ## __VA_ARGS__)

#define LOG_WARN(fmt, ...) \
	Logger::Get().LogFormat(LogLevel::Warning, fmt, ## __VA_ARGS__)

#define LOG_ERROR(fmt, ...) \
	Logger::Get().LogFormat(LogLevel::Error, fmt, ## __VA_ARGS__)

#define LOG_MESSAGE(level, message) \
	Logger::Get().Log(level, message)
#endif // NO_LOGGING


extern void HandleLogFatalError(char const* file, int line, char const* fmt, ...);

#define LOG_FATAL(fmt, ...) \
	HandleLogFatalError(__FILE__, __LINE__, fmt, ## __VA_ARGS__)