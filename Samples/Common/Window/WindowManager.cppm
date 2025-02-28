module;

#include <glfw/glfw3.h>

export module WindowManager;
import vulkan_hpp;

export using ::GLFWwindow;

export class WindowManager {
public:
	static void Init();
	static void Terminate();
	
	static void SetErrorCallback(GLFWerrorfun callback);

	static vk::Result CreateWindowSurface(vk::Instance                   instance,
										  GLFWwindow*                    window,
										  const vk::AllocationCallbacks* allocator,
										  vk::SurfaceKHR*                surface);

	static const char** GetRequiredInstanceExtensions(uint32_t* count);

	static void PollEvents();
	static void WaitEvents();
};
