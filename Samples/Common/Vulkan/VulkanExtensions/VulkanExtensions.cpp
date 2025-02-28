module;

#include <vulkan/vulkan.h>

#include "VulkanExtensionsDefinitions.h"

module VulkanExtensions;

import vulkan_hpp;

void LoadInstanceDebugUtilsFunctionsEXT(vk::Instance instance) {
	pfn_vkCreateDebugUtilsMessengerEXT =
		(PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	pfn_vkDestroyDebugUtilsMessengerEXT = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
		instance, "vkDestroyDebugUtilsMessengerEXT");
}

void LoadDeviceDebugUtilsFunctionsEXT(vk::Device device) {
	pfn_vkSetDebugUtilsObjectNameEXT = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
		vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectName"));
}

void LoadInstanceCooperativeMatrixFunctionsKHR(vk::Instance instance) {
	pfn_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
		vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
}

void LoadInstanceCooperativeMatrix2FunctionsNV(vk::Instance instance) {
	pfn_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV>(
		vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV"));
}

void LoadDeviceCooperativeMatrixFunctionsNV(vk::Device device) {}
