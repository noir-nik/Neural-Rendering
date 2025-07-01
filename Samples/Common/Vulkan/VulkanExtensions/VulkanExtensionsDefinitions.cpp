#include <vulkan/vulkan.h>

#include "VulkanExtensionsDefinitions.h"

import VulkanExtensions;

// VK_EXT_debug_utils
PFN_vkCreateDebugUtilsMessengerEXT  pfn_vkCreateDebugUtilsMessengerEXT  = nullptr;
PFN_vkDestroyDebugUtilsMessengerEXT pfn_vkDestroyDebugUtilsMessengerEXT = nullptr;
PFN_vkSetDebugUtilsObjectNameEXT    pfn_vkSetDebugUtilsObjectNameEXT    = nullptr;

// VK_KHR_cooperative_matrix
PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = nullptr;

// VK_NV_cooperative_matrix2
PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV pfn_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV = nullptr;

// Vulkan VK_NV_cooperative_vector
PFN_vkGetPhysicalDeviceCooperativeVectorPropertiesNV pfn_vkGetPhysicalDeviceCooperativeVectorPropertiesNV = nullptr;
PFN_vkConvertCooperativeVectorMatrixNV               pfn_vkConvertCooperativeVectorMatrixNV               = nullptr;
PFN_vkCmdConvertCooperativeVectorMatrixNV            pfn_vkCmdConvertCooperativeVectorMatrixNV            = nullptr;
 

// VK_EXT_debug_utils
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

// VK_KHR_cooperative_matrix
VKAPI_ATTR VkResult VKAPI_CALL
vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount,
												  VkCooperativeMatrixPropertiesKHR* pProperties) {
	return pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(physicalDevice, pPropertyCount, pProperties);
}

// VK_NV_cooperative_matrix2
VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV(
	VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount,
	VkCooperativeMatrixFlexibleDimensionsPropertiesNV* pProperties) {
	return pfn_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV(
		physicalDevice, pPropertyCount, pProperties);
}

// Vulkan VK_NV_cooperative_vector
VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceCooperativeVectorPropertiesNV(VkPhysicalDevice                 physicalDevice,
																				uint32_t*                        pPropertyCount,
																				VkCooperativeVectorPropertiesNV* pProperties) {
	return pfn_vkGetPhysicalDeviceCooperativeVectorPropertiesNV(physicalDevice, pPropertyCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL vkConvertCooperativeVectorMatrixNV(VkDevice device, const VkConvertCooperativeVectorMatrixInfoNV* pInfo) {
	return pfn_vkConvertCooperativeVectorMatrixNV(device, pInfo);
}

VKAPI_ATTR void VKAPI_CALL vkCmdConvertCooperativeVectorMatrixNV(VkCommandBuffer commandBuffer, uint32_t infoCount, const VkConvertCooperativeVectorMatrixInfoNV* pInfos) {
	pfn_vkCmdConvertCooperativeVectorMatrixNV(commandBuffer, infoCount, pInfos);
}

#if 0
VKAPI_ATTR VkResult VKAPI_CALL vkGetMemoryWin32HandleKHR(
	VkDevice                             device,
	const VkMemoryGetWin32HandleInfoKHR* pGetWin32HandleInfo,
	HANDLE*                              pHandle) {
	return pfn_vkGetMemoryWin32HandleKHR(device, pGetWin32HandleInfo, pHandle);
};

VKAPI_ATTR VkResult VKAPI_CALL vkGetMemoryWin32HandlePropertiesKHR(
	VkDevice                           device,
	VkExternalMemoryHandleTypeFlagBits handleType,
	HANDLE                             handle,
	VkMemoryWin32HandlePropertiesKHR*  pMemoryWin32HandleProperties) {
	return pfn_vkGetMemoryWin32HandlePropertiesKHR(device, handleType, handle, pMemoryWin32HandleProperties);
};
#endif
