module;
#include <GLFW/glfw3.h>
module Window;

import WindowCreateInfo;

static void WindowPosCallback(GLFWwindow* window, int x, int y);
static void WindowSizeCallback(GLFWwindow* window, int width, int height);
static void WindowCloseCallback(GLFWwindow* window);
static void WindowRefreshCallback(GLFWwindow* window);
static void WindowFocusCallback(GLFWwindow* window, int focused);
static void WindowIconifyCallback(GLFWwindow* window, int iconified);
static void WindowMaximizeCallback(GLFWwindow* window, int maximize);
static void FramebufferSizeCallback(GLFWwindow* window, int width, int height);
static void WindowContentScaleCallback(GLFWwindow* window, float xscale, float yscale);

static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
static void CursorPosCallback(GLFWwindow* window, double xpos, double ypos);
static void CursorEnterCallback(GLFWwindow* window, int entered);
static void ScrollCallback(GLFWwindow* window, double x, double y);
static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
static void CharCallback(GLFWwindow* window, unsigned int codepoint);
static void CharModsCallback(GLFWwindow* window, unsigned int codepoint, int mods);
static void DropCallback(GLFWwindow* window, int path_count, char const* paths[]);

GLFWWindow::GLFWWindow() {}

void GLFWWindow::Init(WindowCreateInfo const& info) {
	if (info.mode == WindowMode::eFullscreen) {
		SetWindowMode(WindowMode::eFullscreen);
		auto monitor   = glfwGetPrimaryMonitor();
		auto videoMode = glfwGetVideoMode(monitor);
		glfwWindowHint(GLFW_RED_BITS, videoMode->redBits);
		glfwWindowHint(GLFW_GREEN_BITS, videoMode->greenBits);
		glfwWindowHint(GLFW_BLUE_BITS, videoMode->blueBits);
		glfwWindowHint(GLFW_REFRESH_RATE, videoMode->refreshRate);
	}

	glfwWindowHint(GLFW_RESIZABLE, info.bResizable);
	glfwWindowHint(GLFW_DECORATED, info.bDecorated);
	glfwWindowHint(GLFW_FOCUSED, info.bFocused);
	glfwWindowHint(GLFW_AUTO_ICONIFY, info.bAutoIconify);
	glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, info.bTransparent);
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	window = glfwCreateWindow(info.width, info.height, info.title, nullptr, nullptr);
	glfwSetWindowUserPointer(window, this);
	// Set pos from info
	int x, y;
	glfwGetWindowPos(window, &x, &y);
	if (info.x != kWindowDontCare) {
		x = info.x;
	}
	if (info.y != kWindowDontCare) {
		y = info.y;
	}
	if (x != kWindowDontCare || y != kWindowDontCare) {
		glfwSetWindowPos(window, x, y);
	}

	// Get Stored Rect
	glfwGetWindowPos(window, &windowed_dimensions.x, &windowed_dimensions.y);
	glfwGetWindowSize(window, &windowed_dimensions.width, &windowed_dimensions.height);

	glfwSetWindowPosCallback(window, WindowPosCallback);
	glfwSetWindowSizeCallback(window, WindowSizeCallback);
	glfwSetWindowCloseCallback(window, WindowCloseCallback);
	glfwSetWindowRefreshCallback(window, WindowRefreshCallback);
	glfwSetWindowFocusCallback(window, WindowFocusCallback);
	glfwSetWindowIconifyCallback(window, WindowIconifyCallback);
	glfwSetWindowMaximizeCallback(window, WindowMaximizeCallback);
	glfwSetFramebufferSizeCallback(window, FramebufferSizeCallback);
	glfwSetWindowContentScaleCallback(window, WindowContentScaleCallback);

	glfwSetMouseButtonCallback(window, MouseButtonCallback);
	glfwSetCursorPosCallback(window, CursorPosCallback);
	glfwSetScrollCallback(window, ScrollCallback);
	glfwSetKeyCallback(window, KeyCallback);
	glfwSetCharCallback(window, CharCallback);
	glfwSetCharModsCallback(window, CharModsCallback);
	glfwSetDropCallback(window, DropCallback);
}

GLFWWindow::~GLFWWindow() {
	Destroy();
}

void GLFWWindow::SetPos(int x, int y) {
	glfwSetWindowPos(window, x, y);
}

void GLFWWindow::SetSize(int width, int height) {
	glfwSetWindowSize(window, width, height);
}

bool GLFWWindow::GetFullScreenRect(int& x, int& y, int& width, int& height) const {
	auto monitor   = glfwGetPrimaryMonitor();
	auto videoMode = glfwGetVideoMode(monitor);
	glfwGetMonitorPos(monitor, &x, &y);
	width  = videoMode->width;
	height = videoMode->height;
	return true;
}

void GLFWWindow::SetSizeLimits(int minWidth, int minHeight, int maxWidth, int maxHeight) {
	auto TranslateWindowDontCare = [](int value) {
		if (value == kWindowDontCare) {
			return GLFW_DONT_CARE;
		}
		return value;
	};
	glfwSetWindowSizeLimits(window, TranslateWindowDontCare(minWidth), TranslateWindowDontCare(minHeight),
							TranslateWindowDontCare(maxWidth), TranslateWindowDontCare(maxHeight));
}

void GLFWWindow::Destroy() {
	if (window) {
		glfwDestroyWindow(window);
		window = nullptr;
	}
}

void GLFWWindow::Minimize() {
	glfwIconifyWindow(window);
}

void GLFWWindow::Maximize() {
	glfwMaximizeWindow(window);
}

void GLFWWindow::Restore() {
	glfwRestoreWindow(window);
}

void GLFWWindow::Show() {
	glfwShowWindow(window);
}

void GLFWWindow::Hide() {
	glfwHideWindow(window);
}

void GLFWWindow::SetWindowMode(WindowMode mode) {
	auto monitor   = glfwGetPrimaryMonitor();
	auto videoMode = glfwGetVideoMode(monitor);
	switch (mode) {
	case WindowMode::eWindowed:
		glfwSetWindowMonitor(window, nullptr, windowed_dimensions.x, windowed_dimensions.y, windowed_dimensions.width, windowed_dimensions.height, GLFW_DONT_CARE);
		break;
	case WindowMode::eWindowedFullscreen:
		glfwGetWindowPos(window, &windowed_dimensions.x, &windowed_dimensions.y);
		glfwGetWindowSize(window, &windowed_dimensions.width, &windowed_dimensions.height);
		glfwSetWindowMonitor(window, nullptr, 0, 0, videoMode->width, videoMode->height, GLFW_DONT_CARE);
		break;
	case WindowMode::eFullscreen:
		glfwGetWindowPos(window, &windowed_dimensions.x, &windowed_dimensions.y);
		glfwGetWindowSize(window, &windowed_dimensions.width, &windowed_dimensions.height);
		glfwSetWindowMonitor(window, monitor, 0, 0, videoMode->width, videoMode->height, videoMode->refreshRate);
		break;
	}
}

auto GLFWWindow::GetWindowMode() const -> WindowMode {
	return mode;
}

bool GLFWWindow::IsMaximized() const {
	return glfwGetWindowAttrib(window, GLFW_MAXIMIZED);
}

bool GLFWWindow::IsMinimized() const {
	return glfwGetWindowAttrib(window, GLFW_ICONIFIED);
}

bool GLFWWindow::GetRestoredRect(int& x, int& y, int& width, int& height) {
	x      = windowed_dimensions.x;
	y      = windowed_dimensions.y;
	width  = windowed_dimensions.width;
	height = windowed_dimensions.height;
	return true;
}

void GLFWWindow::GetRect(int& x, int& y, int& width, int& height) {
	glfwGetWindowPos(window, &x, &y);
	glfwGetWindowSize(window, &width, &height);
}

void GLFWWindow::SetFocus() {
	glfwFocusWindow(window);
}

void GLFWWindow::SetOpacity(float const value) {
	glfwSetWindowOpacity(window, value);
}

void GLFWWindow::Enable(bool bEnable) {
	glfwSetInputMode(window, GLFW_CURSOR, bEnable ? GLFW_CURSOR_NORMAL : GLFW_CURSOR_DISABLED);
}

auto GLFWWindow::GetHandle() const -> void* {
	return window;
}

void GLFWWindow::SetText(const char* const text) {
	glfwSetWindowTitle(window, text);
}

bool GLFWWindow::GetShouldClose() const {
	return glfwWindowShouldClose(window);
}

void GLFWWindow::SetShouldClose(bool value) const {
	glfwSetWindowShouldClose(window, value ? GLFW_TRUE : GLFW_FALSE);
}

void GLFWWindow::SetUserPointer(void* user_pointer) {
	this->user_pointer = user_pointer;
}

auto GLFWWindow::GetUserPointer() const -> void* {
	return this->user_pointer;
}

#if 0
#include <cstdio>
#define LOG_WINDOW(fmt, ...) std::printf(fmt "\n", ##__VA_ARGS__);
#define LOG_INPUT(fmt, ...) std::printf(fmt "\n", ##__VA_ARGS__);
#else
#define LOG_WINDOW(fmt, ...)
#define LOG_INPUT(fmt, ...)
#endif

static void WindowPosCallback(GLFWwindow* window, int x, int y) {
	GLFWWindow* pWindow = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
	LOG_WINDOW("Window %p moved to %dx%d", pWindow, x, y);
	if (pWindow->GetWindowCallbacks().windowPosCallback)
		pWindow->GetWindowCallbacks().windowPosCallback(pWindow, x, y);
}

static void WindowSizeCallback(GLFWwindow* window, int width, int height) {
	GLFWWindow* pWindow = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
	LOG_WINDOW("Window %p resized to %dx%d", pWindow, width, height);
	if (pWindow->GetWindowCallbacks().windowSizeCallback)
		pWindow->GetWindowCallbacks().windowSizeCallback(pWindow, width, height);
}

static void WindowCloseCallback(GLFWwindow* window) {
	GLFWWindow* pWindow = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
	LOG_WINDOW("Window %p closed", pWindow);
	if (pWindow->GetWindowCallbacks().windowCloseCallback)
		pWindow->GetWindowCallbacks().windowCloseCallback(pWindow);
}

static void WindowRefreshCallback(GLFWwindow* window) {
	GLFWWindow* pWindow = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
	LOG_WINDOW("Window %p refreshed", pWindow);

	if (pWindow->GetWindowCallbacks().windowRefreshCallback)
		pWindow->GetWindowCallbacks().windowRefreshCallback(pWindow);
}

static void WindowFocusCallback(GLFWwindow* window, int focused) {
	GLFWWindow* pWindow = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
	LOG_WINDOW("Window %p focused %d", pWindow, focused);
	if (pWindow->GetWindowCallbacks().windowFocusCallback)
		pWindow->GetWindowCallbacks().windowFocusCallback(pWindow, focused);
}

static void WindowIconifyCallback(GLFWwindow* window, int iconified) {
	GLFWWindow* pWindow = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
	LOG_WINDOW("Window %p iconified %d", pWindow, iconified);

	if (pWindow->GetWindowCallbacks().windowIconifyCallback)
		pWindow->GetWindowCallbacks().windowIconifyCallback(pWindow, iconified);
}

static void WindowMaximizeCallback(GLFWwindow* window, int maximize) {
	GLFWWindow* pWindow = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
	LOG_WINDOW("Window %p maximized %d", pWindow, maximize);
	if (pWindow->GetWindowCallbacks().windowMaximizeCallback)
		pWindow->GetWindowCallbacks().windowMaximizeCallback(pWindow, maximize);
}

static void FramebufferSizeCallback(GLFWwindow* window, int width, int height) {
	GLFWWindow* pWindow = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
	LOG_WINDOW("Window %p framebuffer resized to %dx%d", pWindow, width, height);
	if (pWindow->GetWindowCallbacks().framebufferSizeCallback)
		pWindow->GetWindowCallbacks().framebufferSizeCallback(pWindow, width, height);
}

static void WindowContentScaleCallback(GLFWwindow* window, float xscale, float yscale) {
	GLFWWindow* pWindow = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
	LOG_WINDOW("Window %p content scale changed to %fx%f", pWindow, xscale, yscale);
	if (pWindow->GetWindowCallbacks().windowContentScaleCallback)
		pWindow->GetWindowCallbacks().windowContentScaleCallback(pWindow, xscale, yscale);
}

static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	GLFWWindow* pWindow = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
	LOG_INPUT("Window %p mouse button: %d, action: %d, mods: %d", pWindow, button, action, mods);
	if (pWindow->GetInputCallbacks().mouseButtonCallback)
		pWindow->GetInputCallbacks().mouseButtonCallback(pWindow, button, action, mods);
}

static void CursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
	GLFWWindow* pWindow = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
	LOG_INPUT("Window %p cursor pos: %lf, %lf", pWindow, xpos, ypos);
	if (pWindow->GetInputCallbacks().cursorPosCallback)
		pWindow->GetInputCallbacks().cursorPosCallback(pWindow, xpos, ypos);
}

static void CursorEnterCallback(GLFWwindow* window, int entered) {
	GLFWWindow* pWindow = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
	LOG_INPUT("Window %p cursor entered: %d", pWindow, entered);
	if (pWindow->GetInputCallbacks().cursorEnterCallback)
		pWindow->GetInputCallbacks().cursorEnterCallback(pWindow, entered);
}

static void ScrollCallback(GLFWwindow* window, double x, double y) {
	GLFWWindow* pWindow = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
	LOG_INPUT("Window %p scroll: %lf, %lf", pWindow, x, y);
	if (pWindow->GetInputCallbacks().scrollCallback)
		pWindow->GetInputCallbacks().scrollCallback(pWindow, x, y);
}

static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	GLFWWindow* pWindow = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
	LOG_INPUT("Window %p key: %d, scancode: %d, action: %d, mods: %d", pWindow, key, scancode, action, mods);
	if (pWindow->GetInputCallbacks().keyCallback)
		pWindow->GetInputCallbacks().keyCallback(pWindow, key, scancode, action, mods);
}

static void CharCallback(GLFWwindow* window, unsigned int codepoint) {
	GLFWWindow* pWindow = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
	LOG_INPUT("Window %p char: %d", pWindow, codepoint);
	if (pWindow->GetInputCallbacks().charCallback)
		pWindow->GetInputCallbacks().charCallback(pWindow, codepoint);
}

static void CharModsCallback(GLFWwindow* window, unsigned int codepoint, int mods) {
	GLFWWindow* pWindow = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
	LOG_INPUT("Window %p char: %d, mods: %d", pWindow, codepoint, mods);
	if (pWindow->GetInputCallbacks().charModsCallback)
		pWindow->GetInputCallbacks().charModsCallback(pWindow, codepoint, mods);
}

static void DropCallback(GLFWwindow* window, int path_count, char const* paths[]) {
	GLFWWindow* pWindow = reinterpret_cast<GLFWWindow*>(glfwGetWindowUserPointer(window));
	LOG_INPUT("Window %p drop: %d", pWindow, path_count);
	if (pWindow->GetInputCallbacks().dropCallback)
		pWindow->GetInputCallbacks().dropCallback(pWindow, path_count, paths);
}
