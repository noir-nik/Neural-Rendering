module;

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

module WindowManager;

void WindowManager::Init() { glfwInit(); }
void WindowManager::Terminate() { glfwTerminate(); }
void WindowManager::PollEvents() { glfwPollEvents(); }
void WindowManager::WaitEvents() { glfwWaitEvents(); }

vk::Result WindowManager::CreateWindowSurface(vk::Instance                   instance,
											  GLFWwindow*                    window,
											  const vk::AllocationCallbacks* allocator,
											  vk::SurfaceKHR*                surface) {

	VkSurfaceKHR vk_surface;
	vk::Result   result = vk::Result(glfwCreateWindowSurface(instance,
															 window,
															 reinterpret_cast<VkAllocationCallbacks const*>(allocator),
															 &vk_surface));
	if (result != vk::Result::eSuccess) {
		return result;
	}
	*surface = vk_surface;
	return result;
}

char const** WindowManager::GetRequiredInstanceExtensions(uint32_t* count) {
	return glfwGetRequiredInstanceExtensions(count);
};

void WindowManager::SetErrorCallback(GLFWerrorfun callback) { glfwSetErrorCallback(callback); }


