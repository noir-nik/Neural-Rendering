module;
#include <cstdio> // stdout, stderr
#include <cstdarg> // va_start, va_end
module NeuralGraphics;
import :Log;
import std;

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

int static constexpr kMessageBufferSize = 1024;

void MessageCallbackDefault(LogLevel level, char const* message) {
	switch (level) {
	[[likely]]
	case LogLevel::Trace:
		std::fprintf(stdout, "%s\n", message);
		break;
	case LogLevel::Info:
		std::fprintf(stdout, "[" ANSI_COLOR_GREEN "I" ANSI_COLOR_RESET "] %s\n", message);
		break;
	case LogLevel::Warning:
		std::fprintf(stdout, "[" ANSI_COLOR_YELLOW "W" ANSI_COLOR_RESET "] %s\n", message);
		break;
	case LogLevel::Error:
		std::fprintf(stderr, "[" ANSI_COLOR_RED "E" ANSI_COLOR_RESET "] %s\n", message);
		break;
	case LogLevel::Fatal:
		std::fprintf(stderr, "[" ANSI_COLOR_RED "FATAL" ANSI_COLOR_RESET "] %s\n", message);
	}
}

Logger& Logger::Get() {
	static Logger logger;
	return logger;
}

void Logger::LogFormat(LogLevel level, char const* format, ...) {
	char buffer[kMessageBufferSize];
	std::va_list args;
	va_start(args, format);
	int size = vsnprintf(buffer, sizeof(buffer) - 1, format, args);
	va_end(args);
	buffer[sizeof(buffer) - 1] = '\0';
	MessageCallbackDefault(level, buffer);
}

void Logger::Log(LogLevel level, char const* message) {
	MessageCallbackDefault(level, message);	
}

void HandleLogFatalError(char const* file, int line, char const* fmt, ...) {
	std::va_list args;
	va_start(args, fmt);
	char buffer[kMessageBufferSize];
	int size = vsnprintf(buffer, sizeof(buffer) - 1, fmt, args);
	size += snprintf(buffer + size, sizeof(buffer) - size - 1, " %s:%d ", file, line);
	va_end(args);
	buffer[sizeof(buffer) - 1] = '\0';
	// todo: Messagebox
	MessageCallbackDefault(LogLevel::Fatal, buffer);
	std::exit(1);
}
