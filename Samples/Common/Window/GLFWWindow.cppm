module;

#include <GLFW/glfw3.h>

export module Window:GLFWWindow;
export import IWindow;
export import WindowCreateInfo;

struct WindowRect {
	int x, y, width, height;
};

export class GLFWWindow;

export struct WindowCallbacks {
	void (*windowPosCallback)(GLFWWindow* window, int xpos, int ypos)                  = nullptr;
	void (*windowSizeCallback)(GLFWWindow* window, int width, int height)              = nullptr;
	void (*windowCloseCallback)(GLFWWindow* window)                                    = nullptr;
	void (*windowRefreshCallback)(GLFWWindow* window)                                  = nullptr;
	void (*windowFocusCallback)(GLFWWindow* window, int focused)                       = nullptr;
	void (*windowIconifyCallback)(GLFWWindow* window, int iconified)                   = nullptr;
	void (*windowMaximizeCallback)(GLFWWindow* window, int maximized)                  = nullptr;
	void (*framebufferSizeCallback)(GLFWWindow* window, int width, int height)         = nullptr;
	void (*windowContentScaleCallback)(GLFWWindow* window, float xscale, float yscale) = nullptr;
};

export struct WindowInputCallbacks {
	void (*mouseButtonCallback)(GLFWWindow* window, int button, int action, int mods)    = nullptr;
	void (*cursorPosCallback)(GLFWWindow* window, double xpos, double ypos)              = nullptr;
	void (*cursorEnterCallback)(GLFWWindow* window, int entered)                         = nullptr;
	void (*scrollCallback)(GLFWWindow* window, double xoffset, double yoffset)           = nullptr;
	void (*keyCallback)(GLFWWindow* window, int key, int scancode, int action, int mods) = nullptr;
	void (*charCallback)(GLFWWindow* window, unsigned int codepoint)                     = nullptr;
	void (*charModsCallback)(GLFWWindow* window, unsigned int codepoint, int mods)       = nullptr;
	void (*dropCallback)(GLFWWindow* window, int path_count, char const* paths[])        = nullptr;
};

export class GLFWWindow : public IWindow {
public:
	GLFWWindow();

	void Init(WindowCreateInfo const& info);

public:
	~GLFWWindow() override;
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
	virtual void SetUserPointer(void* user_pointer) override;
	virtual auto GetUserPointer() const -> void* override;

	auto GetWindowCallbacks() -> WindowCallbacks& { return callbacks; }
	auto GetInputCallbacks() -> WindowInputCallbacks& { return input_callbacks; }

private:
	GLFWwindow* window = nullptr;
	WindowMode  mode   = WindowMode::eWindowed;

	void* user_pointer = nullptr;

	WindowRect windowed_dimensions;

	// GLFW callbacks (no events are exposed)
	WindowCallbacks      callbacks;
	WindowInputCallbacks input_callbacks;
};
