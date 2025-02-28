export module VulkanExtensions;
import vulkan_hpp;

export {
	void LoadInstanceDebugUtilsFunctionsEXT(vk::Instance instance);
	void LoadDeviceDebugUtilsFunctionsEXT(vk::Device device);
	void LoadInstanceCooperativeMatrixFunctionsKHR(vk::Instance instance);
	void LoadInstanceCooperativeMatrix2FunctionsNV(vk::Instance instance);
	void LoadDeviceCooperativeVectorFunctionsNV(vk::Device device);
}
