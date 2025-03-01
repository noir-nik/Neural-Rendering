module MessageCallbacks;

import std;
import vulkan_hpp;

#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_YELLOW "\x1b[33m"
#define ANSI_COLOR_RESET "\x1b[0m"

vk::Bool32 DebugUtilsCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
							  vk::DebugUtilsMessageTypeFlagsEXT             messageTypes,
							  const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
							  void*                                         pUserData) {

	// Silence validation layer not knowing Cooperative Vector extension
	switch (static_cast<unsigned>(pCallbackData->messageIdNumber)) {
	case 0x79de34d4: [[fallthrough]];
	case 0x901f59ec:
		return vk::False;
	}
	if (messageSeverity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose) {
		std::printf("%s\n", pCallbackData->pMessage);
	}
	if (messageSeverity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo) {
		std::printf("[" ANSI_COLOR_GREEN "I" ANSI_COLOR_RESET "] %s\n", pCallbackData->pMessage);
	}
	if (messageSeverity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
		std::printf("[" ANSI_COLOR_YELLOW "W" ANSI_COLOR_RESET "] %s\n", pCallbackData->pMessage);
	}
	if (messageSeverity & vk::DebugUtilsMessageSeverityFlagBitsEXT::eError) {
		std::printf("[" ANSI_COLOR_RED "E" ANSI_COLOR_RESET "] %s\n", pCallbackData->pMessage);
	}
	return vk::False;
}

void WindowErrorCallback(int error, char const* description) {
	std::printf("[" ANSI_COLOR_RED "Window Error" ANSI_COLOR_RESET "] (%d) {%s}", error, description);
}
