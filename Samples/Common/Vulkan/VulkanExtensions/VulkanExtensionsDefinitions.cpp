#include <vulkan/vulkan.h>

#include "VulkanExtensionsDefinitions.h"

import VulkanExtensions;

PFN_vkCreateDebugUtilsMessengerEXT                                     pfn_vkCreateDebugUtilsMessengerEXT                                      = nullptr;
PFN_vkDestroyDebugUtilsMessengerEXT                                    pfn_vkDestroyDebugUtilsMessengerEXT                                     = nullptr;
PFN_vkSetDebugUtilsObjectNameEXT                                       pfn_vkSetDebugUtilsObjectNameEXT                                        = nullptr;
PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR                  pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR                  = nullptr;
PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV pfn_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV = nullptr;

VKAPI_ATTR VkResult VKAPI_CALL vkCreateDebugUtilsMessengerEXT(
	VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
	const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pMessenger) {
	return pfn_vkCreateDebugUtilsMessengerEXT(instance, pCreateInfo, pAllocator, pMessenger);
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDebugUtilsMessengerEXT(VkInstance                   instance,
														   VkDebugUtilsMessengerEXT     messenger,
														   VkAllocationCallbacks const* pAllocator) {
	return pfn_vkDestroyDebugUtilsMessengerEXT(instance, messenger, pAllocator);
}

VKAPI_ATTR VkResult VKAPI_CALL vkSetDebugUtilsObjectNameEXT(VkDevice                             device,
															const VkDebugUtilsObjectNameInfoEXT* pNameInfo) {
	return pfn_vkSetDebugUtilsObjectNameEXT(device, pNameInfo);
}

VKAPI_ATTR VkResult VKAPI_CALL
vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount,
												  VkCooperativeMatrixPropertiesKHR* pProperties) {
	// auto t = pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR;
	return pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(physicalDevice, pPropertyCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV(
	VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount,
	VkCooperativeMatrixFlexibleDimensionsPropertiesNV* pProperties) {
	return pfn_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV(
		physicalDevice, pPropertyCount, pProperties);
}
