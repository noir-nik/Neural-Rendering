export module NeuralGraphics:Log;

export
enum class LogLevel {
	Trace,
	Info,
	Warning,
	Error, 
	Fatal,
};	


export
class Logger {
public:
	// Get the singleton instance of the Logger
	static Logger& Get();

	// Logging function
	void LogFormat(LogLevel level, char const* format, ...);
	void Log(LogLevel level, char const* message);
private:
	Logger() = default;
};

