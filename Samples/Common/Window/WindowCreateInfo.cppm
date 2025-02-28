export module WindowCreateInfo;
import IWindow;

export struct WindowCreateInfo {
	int x      = kWindowDontCare;
	int y      = kWindowDontCare;
	int width  = kWindowDontCare;
	int height = kWindowDontCare;

	WindowMode  mode  = WindowMode::eWindowed;
	char const* title = "Window";

	bool bResizable : 1   = true;
	bool bDecorated : 1   = true;
	bool bFocused : 1     = true;
	bool bAutoIconify : 1 = false;
	bool bTransparent : 1 = false;
};
