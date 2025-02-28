module VulkanRHI;
import :PhysicalDevice;
import vulkan_hpp;
import std;

namespace VulkanRHI {

auto PhysicalDevice::GetDetails() -> vk::Result {
	// get all available extensions
	auto [result, extensions] = enumerateDeviceExtensionProperties();
	if (result != vk::Result::eSuccess)
		return result;
	available_extensions = std::move(extensions);

	// get features, properties and queue families
	getFeatures2(&GetFeatures2());
	getProperties2(&GetProperties2());
	getMemoryProperties2(&memory_properties);
	queue_family_properties = getQueueFamilyProperties2();

	// Get max number of samples
	vk::SampleCountFlags sample_counts = GetProperties10().limits.framebufferColorSampleCounts;
	for (vk::SampleCountFlagBits sampleCount = vk::SampleCountFlagBits::e64; sampleCount >= vk::SampleCountFlagBits::e1;
		 sampleCount                         = static_cast<vk::SampleCountFlagBits>(vk::SampleCountFlags::MaskType(sampleCount) >> 1)) {
		if (sample_counts & sampleCount) {
			max_samples = static_cast<vk::SampleCountFlagBits>(sampleCount);
			break;
		}
	}
	return vk::Result::eSuccess;
}

auto PhysicalDevice::SupportsExtension(std::string_view extension) const -> bool {
	return std::any_of(available_extensions.begin(), available_extensions.end(),
					   [&extension](vk::ExtensionProperties const& available_extension) {
						   return available_extension.extensionName == extension;
					   });
}

auto PhysicalDevice::SupportsExtensions(std::span<char const* const> extensions) const -> bool {
	return std::all_of(extensions.begin(), extensions.end(), [available_extensions = available_extensions](std::string_view extension) {
		return std::any_of(available_extensions.begin(), available_extensions.end(),
						   [&extension](vk::ExtensionProperties const& available_extension) {
							   return available_extension.extensionName == extension;
						   });
	});
}

auto PhysicalDevice::GetQueueFamilyIndex(vk::QueueFlags desired_flags, vk::QueueFlags undesired_flags) const -> std::optional<u32> {
	for (u32 index = 0; index < queue_family_properties.size(); ++index) {
		if ((queue_family_properties[index].queueFamilyProperties.queueFlags & desired_flags) == desired_flags &&
			(undesired_flags == vk::QueueFlags{} ||
			 (queue_family_properties[index].queueFamilyProperties.queueFlags & undesired_flags) != undesired_flags)) {
			return index;
		}
	}
	return std::nullopt;
}

auto PhysicalDevice::GetQueueFamilyIndex(QueueFamilyInfo const& info) const -> std::pair<vk::Result, std::optional<u32>> {
	for (u32 index = 0; index < queue_family_properties.size(); ++index) {
		if ((queue_family_properties[index].queueFamilyProperties.queueFlags & info.flags) == info.flags &&
			(info.undesired_flags == vk::QueueFlags{} ||
			 (queue_family_properties[index].queueFamilyProperties.queueFlags & info.undesired_flags) != info.undesired_flags)) {
			vk::Bool32 bSupportsSurface;
			vk::Result result = getSurfaceSupportKHR(index, info.surface, &bSupportsSurface);
			if (result != vk::Result::eSuccess) {
				return {result, std::nullopt};
			}
			if (bSupportsSurface) {
				return {vk::Result::eSuccess, index};
			}
		}
	}
	return {vk::Result::eSuccess, std::nullopt};
}

bool PhysicalDevice::SupportsQueue(QueueFamilyInfo const& queue_info) const {
	auto [result, index] = GetQueueFamilyIndex(queue_info);
	return result == vk::Result::eSuccess && index.has_value();
};

bool PhysicalDevice::SupportsQueues(std::span<QueueFamilyInfo const> queue_infos) const {
	for (auto const& queue_info : queue_infos) {
		if (!SupportsQueue(queue_info)) {
			return false;
		}
	}
	return true;
};

auto PhysicalDevice::GetDedicatedTransferQueueFamilyIndex() const -> std::optional<u32> {
	for (u32 index = 0; index < queue_family_properties.size(); ++index) {
		if (((queue_family_properties[index].queueFamilyProperties.queueFlags & vk::QueueFlagBits::eTransfer) == vk::QueueFlagBits::eTransfer) &&
			!((queue_family_properties[index].queueFamilyProperties.queueFlags & vk::QueueFlagBits::eGraphics) == vk::QueueFlagBits::eGraphics) &&
			!((queue_family_properties[index].queueFamilyProperties.queueFlags & vk::QueueFlagBits::eCompute) == vk::QueueFlagBits::eCompute)) {
			return index;
		}
	}
	return std::nullopt;
}

auto PhysicalDevice::GetDedicatedComputeQueueFamilyIndex() const -> std::optional<u32> {
	for (u32 index = 0; index < queue_family_properties.size(); ++index) {
		if (((queue_family_properties[index].queueFamilyProperties.queueFlags & vk::QueueFlagBits::eCompute) == vk::QueueFlagBits::eCompute) &&
			!((queue_family_properties[index].queueFamilyProperties.queueFlags & vk::QueueFlagBits::eGraphics) == vk::QueueFlagBits::eGraphics)) {
			return index;
		}
	}
	return std::nullopt;
}

auto PhysicalDevice::GetQueueCount(u32 queue_family_index) const -> u32 {
	return queue_family_properties[queue_family_index].queueFamilyProperties.queueCount;
}

auto PhysicalDevice::GetQueueFamilyProperties(u32 queue_family_index) const -> vk::QueueFamilyProperties const& {
	return queue_family_properties[queue_family_index].queueFamilyProperties;
}

auto PhysicalDevice::GetMaxPushConstantsSize() const -> u32 {
	return GetProperties10().limits.maxPushConstantsSize;
}

} // namespace VulkanRHI
