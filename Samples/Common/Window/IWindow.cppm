export module IWindow;

export enum class WindowMode {
	eWindowed,
	eWindowedFullscreen,
	eFullscreen,
};

export constexpr inline auto WindowModeToString(WindowMode mode) -> const char* {
	switch (mode) {
		case WindowMode::eWindowed: return "Windowed";
		case WindowMode::eWindowedFullscreen: return "WindowedFullscreen";
		case WindowMode::eFullscreen: return "Fullscreen";
		default: return "Unknown";
	}
}

export constexpr inline int kWindowDontCare = -1;

export class IWindow {
  public:
	virtual ~IWindow() {}

	virtual void SetPos(int x, int y) = 0;
	virtual void SetSize(int width, int height) = 0;
	virtual bool GetFullScreenRect(int& x, int& y, int& width, int& height) const = 0;
	virtual void SetSizeLimits(int min_width, int min_height, int max_width, int max_height) = 0;
	virtual void Destroy() = 0;
	virtual void Minimize() = 0;
	virtual void Maximize() = 0;
	virtual void Restore() = 0;
	virtual void Show() = 0;
	virtual void Hide() = 0;
	virtual void SetWindowMode(WindowMode mode) = 0;
	virtual auto GetWindowMode() const -> WindowMode = 0;
	virtual bool IsMaximized() const = 0;
	virtual bool IsMinimized() const = 0;
	virtual bool GetRestoredRect(int& x, int& y, int& width, int& height) = 0;
	virtual void GetRect(int& x, int& y, int& width, int& height) = 0;
	virtual void SetFocus() = 0;
	virtual void SetOpacity(float const value) = 0;
	virtual void Enable(bool bEnable) = 0;
	virtual auto GetHandle() const -> void* = 0;
	virtual void SetText(const char* const text) = 0;

	virtual void SetUserPointer(void* user_pointer) = 0;
	virtual auto GetUserPointer() const -> void* = 0;
};