module;

#include <GLFW/glfw3.h>

export module Window;
export import IWindow;
export import WindowCreateInfo;

struct WindowRect {
	int x, y, width, height;
};

export class Window;

export struct WindowCallbacks {
	void (*windowPosCallback)(Window* window, int xpos, int ypos)                  = nullptr;
	void (*windowSizeCallback)(Window* window, int width, int height)              = nullptr;
	void (*windowCloseCallback)(Window* window)                                    = nullptr;
	void (*windowRefreshCallback)(Window* window)                                  = nullptr;
	void (*windowFocusCallback)(Window* window, int focused)                       = nullptr;
	void (*windowIconifyCallback)(Window* window, int iconified)                   = nullptr;
	void (*windowMaximizeCallback)(Window* window, int maximized)                  = nullptr;
	void (*framebufferSizeCallback)(Window* window, int width, int height)         = nullptr;
	void (*windowContentScaleCallback)(Window* window, float xscale, float yscale) = nullptr;
};

export struct WindowInputCallbacks {
	void (*mouseButtonCallback)(Window* window, int button, int action, int mods)    = nullptr;
	void (*cursorPosCallback)(Window* window, double xpos, double ypos)              = nullptr;
	void (*cursorEnterCallback)(Window* window, int entered)                         = nullptr;
	void (*scrollCallback)(Window* window, double xoffset, double yoffset)           = nullptr;
	void (*keyCallback)(Window* window, int key, int scancode, int action, int mods) = nullptr;
	void (*charCallback)(Window* window, unsigned int codepoint)                     = nullptr;
	void (*charModsCallback)(Window* window, unsigned int codepoint, int mods)       = nullptr;
	void (*dropCallback)(Window* window, int path_count, char const* paths[])        = nullptr;
};

export class Window : public IWindow {
public:
	Window();

	void Init(WindowCreateInfo const& info);

public:
	~Window() override;
	virtual void SetPos(int x, int y) override;
	virtual void SetSize(int width, int height) override;
	virtual void SetSizeLimits(int minWidth, int minHeight, int maxWidth, int maxHeight) override;
	virtual void Destroy() override;
	virtual void Minimize() override;
	virtual void Maximize() override;
	virtual void Restore() override;
	virtual void Show() override;
	virtual void Hide() override;
	virtual void SetWindowMode(WindowMode mode) override;
	virtual auto GetWindowMode() const -> WindowMode override;
	virtual bool IsMaximized() const override;
	virtual bool IsMinimized() const override;
	virtual bool GetFullScreenRect(int& x, int& y, int& width, int& height) const override;
	virtual bool GetRestoredRect(int& x, int& y, int& width, int& height) override;
	virtual void GetRect(int& x, int& y, int& width, int& height) override;
	virtual void SetFocus() override;
	virtual void SetOpacity(float const value) override;
	virtual void Enable(bool bEnable) override;
	virtual auto GetHandle() const -> void* override;
	virtual void SetText(const char* const text) override;
	virtual bool GetShouldClose() const;
	virtual void SetShouldClose(bool value) const;

	auto GetWindowCallbacks() -> WindowCallbacks& { return callbacks; }
	auto GetInputCallbacks() -> WindowInputCallbacks& { return inputCallbacks; }

private:
	GLFWwindow* window = nullptr;
	WindowMode  mode   = WindowMode::eWindowed;

	WindowRect windowedDimensions;

	// GLFW callbacks (no events are exposed)
	WindowCallbacks      callbacks;
	WindowInputCallbacks inputCallbacks;
};
