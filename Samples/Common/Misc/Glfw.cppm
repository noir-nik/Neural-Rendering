module;

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

export module Glfw;

export {
	using ::GLFWallocator;
	using ::GLFWcursor;
	using ::GLFWgamepadstate;
	using ::GLFWgammaramp;
	using ::GLFWimage;
	using ::GLFWmonitor;
	using ::GLFWvidmode;
	using ::GLFWwindow;

	using ::GLFWallocatefun;
	using ::GLFWcharfun;
	using ::GLFWcharmodsfun;
	using ::GLFWcursorenterfun;
	using ::GLFWcursorposfun;
	using ::GLFWdeallocatefun;
	using ::GLFWdropfun;
	using ::GLFWerrorfun;
	using ::GLFWframebuffersizefun;
	using ::GLFWjoystickfun;
	using ::GLFWkeyfun;
	using ::GLFWmonitorfun;
	using ::GLFWmousebuttonfun;
	using ::GLFWreallocatefun;
	using ::GLFWscrollfun;
	using ::GLFWwindowclosefun;
	using ::GLFWwindowcontentscalefun;
	using ::GLFWwindowfocusfun;
	using ::GLFWwindowiconifyfun;
	using ::GLFWwindowmaximizefun;
	using ::GLFWwindowposfun;
	using ::GLFWwindowrefreshfun;
	using ::GLFWwindowsizefun;
}

export {
	// NOLINTBEGIN(misc-unused-using-decls)
	using ::glfwInit;
	using ::glfwInitAllocator;
	using ::glfwInitHint;
	using ::glfwTerminate;
#if defined(VK_VERSION_1_0)
	using ::glfwInitVulkanLoader;
#endif
	using ::glfwCreateCursor;
	using ::glfwCreateStandardCursor;
	using ::glfwCreateWindow;
	using ::glfwDefaultWindowHints;
	using ::glfwDestroyCursor;
	using ::glfwDestroyWindow;
	using ::glfwExtensionSupported;
	using ::glfwFocusWindow;
	using ::glfwGetClipboardString;
	using ::glfwGetCurrentContext;
	using ::glfwGetCursorPos;
	using ::glfwGetError;
	using ::glfwGetFramebufferSize;
	using ::glfwGetGamepadName;
	using ::glfwGetGamepadState;
	using ::glfwGetGammaRamp;
	using ::glfwGetInputMode;
	using ::glfwGetJoystickAxes;
	using ::glfwGetJoystickButtons;
	using ::glfwGetJoystickGUID;
	using ::glfwGetJoystickHats;
	using ::glfwGetJoystickName;
	using ::glfwGetJoystickUserPointer;
	using ::glfwGetKey;
	using ::glfwGetKeyName;
	using ::glfwGetKeyScancode;
	using ::glfwGetMonitorContentScale;
	using ::glfwGetMonitorName;
	using ::glfwGetMonitorPhysicalSize;
	using ::glfwGetMonitorPos;
	using ::glfwGetMonitors;
	using ::glfwGetMonitorUserPointer;
	using ::glfwGetMonitorWorkarea;
	using ::glfwGetMouseButton;
	using ::glfwGetPlatform;
	using ::glfwGetPrimaryMonitor;
	using ::glfwGetProcAddress;
	using ::glfwGetRequiredInstanceExtensions;
	using ::glfwGetTime;
	using ::glfwGetTimerFrequency;
	using ::glfwGetTimerValue;
	using ::glfwGetVersion;
	using ::glfwGetVersionString;
	using ::glfwGetVideoMode;
	using ::glfwGetVideoModes;
	using ::glfwGetWindowAttrib;
	using ::glfwGetWindowContentScale;
	using ::glfwGetWindowFrameSize;
	using ::glfwGetWindowMonitor;
	using ::glfwGetWindowOpacity;
	using ::glfwGetWindowPos;
	using ::glfwGetWindowSize;
	using ::glfwGetWindowTitle;
	using ::glfwGetWindowUserPointer;
	using ::glfwHideWindow;
	using ::glfwIconifyWindow;
	using ::glfwJoystickIsGamepad;
	using ::glfwJoystickPresent;
	using ::glfwMakeContextCurrent;
	using ::glfwMaximizeWindow;
	using ::glfwPlatformSupported;
	using ::glfwPollEvents;
	using ::glfwPostEmptyEvent;
	using ::glfwRawMouseMotionSupported;
	using ::glfwRequestWindowAttention;
	using ::glfwRestoreWindow;
	using ::glfwSetCharCallback;
	using ::glfwSetCharModsCallback;
	using ::glfwSetClipboardString;
	using ::glfwSetCursor;
	using ::glfwSetCursorEnterCallback;
	using ::glfwSetCursorPos;
	using ::glfwSetCursorPosCallback;
	using ::glfwSetDropCallback;
	using ::glfwSetErrorCallback;
	using ::glfwSetFramebufferSizeCallback;
	using ::glfwSetGamma;
	using ::glfwSetGammaRamp;
	using ::glfwSetInputMode;
	using ::glfwSetJoystickCallback;
	using ::glfwSetJoystickUserPointer;
	using ::glfwSetKeyCallback;
	using ::glfwSetMonitorCallback;
	using ::glfwSetMonitorUserPointer;
	using ::glfwSetMouseButtonCallback;
	using ::glfwSetScrollCallback;
	using ::glfwSetTime;
	using ::glfwSetWindowAspectRatio;
	using ::glfwSetWindowAttrib;
	using ::glfwSetWindowCloseCallback;
	using ::glfwSetWindowContentScaleCallback;
	using ::glfwSetWindowFocusCallback;
	using ::glfwSetWindowIcon;
	using ::glfwSetWindowIconifyCallback;
	using ::glfwSetWindowMaximizeCallback;
	using ::glfwSetWindowMonitor;
	using ::glfwSetWindowOpacity;
	using ::glfwSetWindowPos;
	using ::glfwSetWindowPosCallback;
	using ::glfwSetWindowRefreshCallback;
	using ::glfwSetWindowShouldClose;
	using ::glfwSetWindowSize;
	using ::glfwSetWindowSizeCallback;
	using ::glfwSetWindowSizeLimits;
	using ::glfwSetWindowTitle;
	using ::glfwSetWindowUserPointer;
	using ::glfwShowWindow;
	using ::glfwSwapBuffers;
	using ::glfwSwapInterval;
	using ::glfwUpdateGamepadMappings;
	using ::glfwVulkanSupported;
	using ::glfwWaitEvents;
	using ::glfwWaitEventsTimeout;
	using ::glfwWindowHint;
	using ::glfwWindowHintString;
	using ::glfwWindowShouldClose;
#if defined(VK_VERSION_1_0)
	using ::glfwCreateWindowSurface;
	using ::glfwGetInstanceProcAddress;
	using ::glfwGetPhysicalDevicePresentationSupport;
#endif // VK_VERSION_1_0
	   // NOLINTEND(misc-unused-using-decls)
}

export namespace Glfw {
enum class Action {
	eRelease = GLFW_RELEASE,
	ePress   = GLFW_PRESS,
	eRepeat  = GLFW_REPEAT,
};

enum class Key {
	eUnknown = GLFW_KEY_UNKNOWN,
	/* Printable keys */
	eSpace        = GLFW_KEY_SPACE,
	eApostrophe   = GLFW_KEY_APOSTROPHE,
	eComma        = GLFW_KEY_COMMA,
	eMinus        = GLFW_KEY_MINUS,
	ePeriod       = GLFW_KEY_PERIOD,
	eSlash        = GLFW_KEY_SLASH,
	e0            = GLFW_KEY_0,
	e1            = GLFW_KEY_1,
	e2            = GLFW_KEY_2,
	e3            = GLFW_KEY_3,
	e4            = GLFW_KEY_4,
	e5            = GLFW_KEY_5,
	e6            = GLFW_KEY_6,
	e7            = GLFW_KEY_7,
	e8            = GLFW_KEY_8,
	e9            = GLFW_KEY_9,
	eSemicolon    = GLFW_KEY_SEMICOLON,
	eEqual        = GLFW_KEY_EQUAL,
	eA            = GLFW_KEY_A,
	eB            = GLFW_KEY_B,
	eC            = GLFW_KEY_C,
	eD            = GLFW_KEY_D,
	eE            = GLFW_KEY_E,
	eF            = GLFW_KEY_F,
	eG            = GLFW_KEY_G,
	eH            = GLFW_KEY_H,
	eI            = GLFW_KEY_I,
	eJ            = GLFW_KEY_J,
	eK            = GLFW_KEY_K,
	eL            = GLFW_KEY_L,
	eM            = GLFW_KEY_M,
	eN            = GLFW_KEY_N,
	eO            = GLFW_KEY_O,
	eP            = GLFW_KEY_P,
	eQ            = GLFW_KEY_Q,
	eR            = GLFW_KEY_R,
	eS            = GLFW_KEY_S,
	eT            = GLFW_KEY_T,
	eU            = GLFW_KEY_U,
	eV            = GLFW_KEY_V,
	eW            = GLFW_KEY_W,
	eX            = GLFW_KEY_X,
	eY            = GLFW_KEY_Y,
	eZ            = GLFW_KEY_Z,
	eLeftBracket  = GLFW_KEY_LEFT_BRACKET,
	eBackslash    = GLFW_KEY_BACKSLASH,
	eRightBracket = GLFW_KEY_RIGHT_BRACKET,
	eGraveAccent  = GLFW_KEY_GRAVE_ACCENT,
	eWorld1       = GLFW_KEY_WORLD_1,
	eWorld2       = GLFW_KEY_WORLD_2,
	/* Function keys */
	eEscape       = GLFW_KEY_ESCAPE,
	eEnter        = GLFW_KEY_ENTER,
	eTab          = GLFW_KEY_TAB,
	eBackspace    = GLFW_KEY_BACKSPACE,
	eInsert       = GLFW_KEY_INSERT,
	eDelete       = GLFW_KEY_DELETE,
	eRight        = GLFW_KEY_RIGHT,
	eLeft         = GLFW_KEY_LEFT,
	eDown         = GLFW_KEY_DOWN,
	eUp           = GLFW_KEY_UP,
	ePageUp       = GLFW_KEY_PAGE_UP,
	ePageDown     = GLFW_KEY_PAGE_DOWN,
	eHome         = GLFW_KEY_HOME,
	eEnd          = GLFW_KEY_END,
	eCapsLock     = GLFW_KEY_CAPS_LOCK,
	eScrollLock   = GLFW_KEY_SCROLL_LOCK,
	eNumLock      = GLFW_KEY_NUM_LOCK,
	ePrintScreen  = GLFW_KEY_PRINT_SCREEN,
	ePause        = GLFW_KEY_PAUSE,
	eF1           = GLFW_KEY_F1,
	eF2           = GLFW_KEY_F2,
	eF3           = GLFW_KEY_F3,
	eF4           = GLFW_KEY_F4,
	eF5           = GLFW_KEY_F5,
	eF6           = GLFW_KEY_F6,
	eF7           = GLFW_KEY_F7,
	eF8           = GLFW_KEY_F8,
	eF9           = GLFW_KEY_F9,
	eF10          = GLFW_KEY_F10,
	eF11          = GLFW_KEY_F11,
	eF12          = GLFW_KEY_F12,
	eF13          = GLFW_KEY_F13,
	eF14          = GLFW_KEY_F14,
	eF15          = GLFW_KEY_F15,
	eF16          = GLFW_KEY_F16,
	eF17          = GLFW_KEY_F17,
	eF18          = GLFW_KEY_F18,
	eF19          = GLFW_KEY_F19,
	eF20          = GLFW_KEY_F20,
	eF21          = GLFW_KEY_F21,
	eF22          = GLFW_KEY_F22,
	eF23          = GLFW_KEY_F23,
	eF24          = GLFW_KEY_F24,
	eF25          = GLFW_KEY_F25,
	eKP0          = GLFW_KEY_KP_0,
	eKP1          = GLFW_KEY_KP_1,
	eKP2          = GLFW_KEY_KP_2,
	eKP3          = GLFW_KEY_KP_3,
	eKP4          = GLFW_KEY_KP_4,
	eKP5          = GLFW_KEY_KP_5,
	eKP6          = GLFW_KEY_KP_6,
	eKP7          = GLFW_KEY_KP_7,
	eKP8          = GLFW_KEY_KP_8,
	eKP9          = GLFW_KEY_KP_9,
	eKPDecimal    = GLFW_KEY_KP_DECIMAL,
	eKPDivide     = GLFW_KEY_KP_DIVIDE,
	eKPMultiply   = GLFW_KEY_KP_MULTIPLY,
	eKPSubtract   = GLFW_KEY_KP_SUBTRACT,
	eKPAdd        = GLFW_KEY_KP_ADD,
	eKPEnter      = GLFW_KEY_KP_ENTER,
	eKPEqual      = GLFW_KEY_KP_EQUAL,
	eLeftShift    = GLFW_KEY_LEFT_SHIFT,
	eLeftControl  = GLFW_KEY_LEFT_CONTROL,
	eLeftAlt      = GLFW_KEY_LEFT_ALT,
	eLeftSuper    = GLFW_KEY_LEFT_SUPER,
	eRightShift   = GLFW_KEY_RIGHT_SHIFT,
	eRightControl = GLFW_KEY_RIGHT_CONTROL,
	eRightAlt     = GLFW_KEY_RIGHT_ALT,
	eRightSuper   = GLFW_KEY_RIGHT_SUPER,
	eMenu         = GLFW_KEY_MENU,
	eLast         = GLFW_KEY_LAST
};

enum class Mod {
	eShift    = GLFW_MOD_SHIFT,
	eControl  = GLFW_MOD_CONTROL,
	eAlt      = GLFW_MOD_ALT,
	eSuper    = GLFW_MOD_SUPER,
	eCapsLock = GLFW_MOD_CAPS_LOCK,
	eNumLock  = GLFW_MOD_NUM_LOCK,
};

inline Mod  operator~(Mod rhs) { return static_cast<Mod>(~static_cast<int>(rhs)); }
inline Mod  operator|(Mod lhs, Mod rhs) { return static_cast<Mod>(static_cast<int>(lhs) | static_cast<int>(rhs)); }
inline Mod& operator|=(Mod& lhs, Mod rhs) { return lhs = lhs | rhs; }
inline Mod  operator&(Mod lhs, Mod rhs) { return static_cast<Mod>(static_cast<int>(lhs) & static_cast<int>(rhs)); }
inline Mod& operator&=(Mod& lhs, Mod rhs) { return lhs = lhs & rhs; }
inline Mod  operator^(Mod lhs, Mod rhs) { return static_cast<Mod>(static_cast<int>(lhs) ^ static_cast<int>(rhs)); }
inline Mod& operator^=(Mod& lhs, Mod rhs) { return lhs = lhs ^ rhs; }

enum class Hat {
	eCentered  = GLFW_HAT_CENTERED,
	eUp        = GLFW_HAT_UP,
	eRight     = GLFW_HAT_RIGHT,
	eDown      = GLFW_HAT_DOWN,
	eLeft      = GLFW_HAT_LEFT,
	eRightUp   = GLFW_HAT_RIGHT_UP,
	eRightDown = GLFW_HAT_RIGHT_DOWN,
	eLeftUp    = GLFW_HAT_LEFT_UP,
	eLeftDown  = GLFW_HAT_LEFT_DOWN,
};

enum class CursorStyle {
	eArrow        = GLFW_ARROW_CURSOR,
	eIbeam        = GLFW_IBEAM_CURSOR,
	eCrosshair    = GLFW_CROSSHAIR_CURSOR,
	ePointingHand = GLFW_POINTING_HAND_CURSOR,
	eResizeEW     = GLFW_RESIZE_EW_CURSOR,
	eResizeNS     = GLFW_RESIZE_NS_CURSOR,
	eResizeNWSE   = GLFW_RESIZE_NWSE_CURSOR,
	eResizeNESW   = GLFW_RESIZE_NESW_CURSOR,
	eResizeAll    = GLFW_RESIZE_ALL_CURSOR,
	eNotAllowed   = GLFW_NOT_ALLOWED_CURSOR,
	eHResize      = GLFW_HRESIZE_CURSOR,
	eVResize      = GLFW_VRESIZE_CURSOR,
	eHand         = GLFW_HAND_CURSOR,
};

enum class Joystick {
	e1    = GLFW_JOYSTICK_1,
	e2    = GLFW_JOYSTICK_2,
	e3    = GLFW_JOYSTICK_3,
	e4    = GLFW_JOYSTICK_4,
	e5    = GLFW_JOYSTICK_5,
	e6    = GLFW_JOYSTICK_6,
	e7    = GLFW_JOYSTICK_7,
	e8    = GLFW_JOYSTICK_8,
	e9    = GLFW_JOYSTICK_9,
	e10   = GLFW_JOYSTICK_10,
	e11   = GLFW_JOYSTICK_11,
	e12   = GLFW_JOYSTICK_12,
	e13   = GLFW_JOYSTICK_13,
	e14   = GLFW_JOYSTICK_14,
	e15   = GLFW_JOYSTICK_15,
	e16   = GLFW_JOYSTICK_16,
	eLast = GLFW_JOYSTICK_LAST,
};

enum class GamepadButton{
	eA           = GLFW_GAMEPAD_BUTTON_A,
	eB           = GLFW_GAMEPAD_BUTTON_B,
	eX           = GLFW_GAMEPAD_BUTTON_X,
	eY           = GLFW_GAMEPAD_BUTTON_Y,
	eLeftBumper  = GLFW_GAMEPAD_BUTTON_LEFT_BUMPER,
	eRightBumper = GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER,
	eBack        = GLFW_GAMEPAD_BUTTON_BACK,
	eStart       = GLFW_GAMEPAD_BUTTON_START,
	eGuide       = GLFW_GAMEPAD_BUTTON_GUIDE,
	eLeftThumb   = GLFW_GAMEPAD_BUTTON_LEFT_THUMB,
	eRightThumb  = GLFW_GAMEPAD_BUTTON_RIGHT_THUMB,
	eDpadUp      = GLFW_GAMEPAD_BUTTON_DPAD_UP,
	eDpadRight   = GLFW_GAMEPAD_BUTTON_DPAD_RIGHT,
	eDpadDown    = GLFW_GAMEPAD_BUTTON_DPAD_DOWN,
	eDpadLeft    = GLFW_GAMEPAD_BUTTON_DPAD_LEFT,
	eLast        = GLFW_GAMEPAD_BUTTON_LAST,
	eCross       = GLFW_GAMEPAD_BUTTON_CROSS,
	eCircle      = GLFW_GAMEPAD_BUTTON_CIRCLE,
	eSquare      = GLFW_GAMEPAD_BUTTON_SQUARE,
	eTriangle    = GLFW_GAMEPAD_BUTTON_TRIANGLE,
};

enum class GamepadAxis {
	eLeftX        = GLFW_GAMEPAD_AXIS_LEFT_X,
	eLeftY        = GLFW_GAMEPAD_AXIS_LEFT_Y,
	eRightX       = GLFW_GAMEPAD_AXIS_RIGHT_X,
	eRightY       = GLFW_GAMEPAD_AXIS_RIGHT_Y,
	eLeftTrigger  = GLFW_GAMEPAD_AXIS_LEFT_TRIGGER,
	eRightTrigger = GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER,
	eLast         = GLFW_GAMEPAD_AXIS_LAST,
};

enum class MouseButton {
	eButton1 = GLFW_MOUSE_BUTTON_1,
	eButton2 = GLFW_MOUSE_BUTTON_2,
	eButton3 = GLFW_MOUSE_BUTTON_3,
	eButton4 = GLFW_MOUSE_BUTTON_4,
	eButton5 = GLFW_MOUSE_BUTTON_5,
	eButton6 = GLFW_MOUSE_BUTTON_6,
	eButton7 = GLFW_MOUSE_BUTTON_7,
	eButton8 = GLFW_MOUSE_BUTTON_8,
	eLast    = GLFW_MOUSE_BUTTON_LAST,
	eLeft    = GLFW_MOUSE_BUTTON_LEFT,
	eRight   = GLFW_MOUSE_BUTTON_RIGHT,
	eMiddle  = GLFW_MOUSE_BUTTON_MIDDLE,
};

enum class WindowHint {
	eFocused                = GLFW_FOCUSED,
	eResizable              = GLFW_RESIZABLE,
	eVisible                = GLFW_VISIBLE,
	eDecorated              = GLFW_DECORATED,
	eAutoIconify            = GLFW_AUTO_ICONIFY,
	eFloating               = GLFW_FLOATING,
	eMaximized              = GLFW_MAXIMIZED,
	eCenterCursor           = GLFW_CENTER_CURSOR,
	eTransparentFramebuffer = GLFW_TRANSPARENT_FRAMEBUFFER,
	eFocusOnShow            = GLFW_FOCUS_ON_SHOW,
	eMousePassthrough       = GLFW_MOUSE_PASSTHROUGH,
	ePositionX              = GLFW_POSITION_X,
	ePositionY              = GLFW_POSITION_Y,
};

enum class WindowAttribute {
	eFocused                = GLFW_FOCUSED,
	eIconified              = GLFW_ICONIFIED,
	eResizable              = GLFW_RESIZABLE,
	eVisible                = GLFW_VISIBLE,
	eDecorated              = GLFW_DECORATED,
	eAutoIconify            = GLFW_AUTO_ICONIFY,
	eFloating               = GLFW_FLOATING,
	eMaximized              = GLFW_MAXIMIZED,
	eTransparentFramebuffer = GLFW_TRANSPARENT_FRAMEBUFFER,
	eHovered                = GLFW_HOVERED,
	eFocusOnShow            = GLFW_FOCUS_ON_SHOW,
	eMousePassthrough       = GLFW_MOUSE_PASSTHROUGH,
};

enum class FramebufferHint {
	eRedBits        = GLFW_RED_BITS,
	eGreenBits      = GLFW_GREEN_BITS,
	eBlueBits       = GLFW_BLUE_BITS,
	eAlphaBits      = GLFW_ALPHA_BITS,
	eDepthBits      = GLFW_DEPTH_BITS,
	eStencilBits    = GLFW_STENCIL_BITS,
	eAccumRedBits   = GLFW_ACCUM_RED_BITS,
	eAccumGreenBits = GLFW_ACCUM_GREEN_BITS,
	eAccumBlueBits  = GLFW_ACCUM_BLUE_BITS,
	eAccumAlphaBits = GLFW_ACCUM_ALPHA_BITS,
	eAuxBuffers     = GLFW_AUX_BUFFERS,
	eStereo         = GLFW_STEREO,
	eSamples        = GLFW_SAMPLES,
	eSrgbCapable    = GLFW_SRGB_CAPABLE,
	eRefreshRate    = GLFW_REFRESH_RATE,
	eDoubleBuffer   = GLFW_DOUBLEBUFFER,
};

enum class FramebufferAttribute {
	eDoubleBuffer = GLFW_DOUBLEBUFFER,
};

enum class ClientApiHint {
	eClientApi              = GLFW_CLIENT_API,
	eContextVersionMajor    = GLFW_CONTEXT_VERSION_MAJOR,
	eContextVersionMinor    = GLFW_CONTEXT_VERSION_MINOR,
	eContextRevision        = GLFW_CONTEXT_REVISION,
	eContextRobustness      = GLFW_CONTEXT_ROBUSTNESS,
	eOpenGLForwardCompat    = GLFW_OPENGL_FORWARD_COMPAT,
	eContextDebug           = GLFW_CONTEXT_DEBUG,
	eOpenGLDebugContext     = GLFW_CONTEXT_DEBUG,
	eOpenGLProfile          = GLFW_OPENGL_PROFILE,
	eContextReleaseBehavior = GLFW_CONTEXT_RELEASE_BEHAVIOR,
	eContextNoError         = GLFW_CONTEXT_NO_ERROR,
	eContextCreationApi     = GLFW_CONTEXT_CREATION_API,
	eScaleToMonitor         = GLFW_SCALE_TO_MONITOR,
	eScaleFramebuffer       = GLFW_SCALE_FRAMEBUFFER,
	eCocoaRetinaFramebuffer = GLFW_COCOA_RETINA_FRAMEBUFFER,
	eCocoaFrameName         = GLFW_COCOA_FRAME_NAME,
	eCocoaGraphicsSwitching = GLFW_COCOA_GRAPHICS_SWITCHING,
	eX11ClassName           = GLFW_X11_CLASS_NAME,
	eX11InstanceName        = GLFW_X11_INSTANCE_NAME,
	eWin32KeyboardMenu      = GLFW_WIN32_KEYBOARD_MENU,
	eWin32ShowDefault       = GLFW_WIN32_SHOWDEFAULT,
	eWaylandAppId           = GLFW_WAYLAND_APP_ID,
};

constexpr inline auto kTrue  = GLFW_TRUE;
constexpr inline auto kFalse = GLFW_FALSE;

} // namespace Glfw
