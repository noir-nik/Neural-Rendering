export module MessageCallbacks;

import std;
import vulkan_hpp;

export {
	vk::Bool32 DebugUtilsCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
								  vk::DebugUtilsMessageTypeFlagsEXT             messageTypes,
								  const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
								  void*                                         pUserData);

	void WindowErrorCallback(int error, char const* description);
}
