export module SamplesCommon:PhysicalDevice;
import VulkanRHI;
import NeuralGraphics;
import vulkan_hpp;
import std;

export struct PhysicalDevice : public VulkanRHI::PhysicalDevice {
	PhysicalDevice() {
		Utils::AddToPNext(GetFeatures2(), cooperative_vector_features);
		Utils::AddToPNext(GetFeatures2(), shader_replicated_composites_features);
		Utils::AddToPNext(GetProperties2(), cooperative_vector_properties);
	}
	bool IsSuitable(vk::SurfaceKHR const& surface, std::span<char const* const> extensions) {
		bool const bSupportsExtensions =
			SupportsExtensions(extensions);
		bool const bSupportsQueues =
			SupportsQueue({.flags = vk::QueueFlagBits::eGraphics, .surface = surface});
		bool const bSupportsCooperativeVector =
			cooperative_vector_features.cooperativeVector == vk::True;
		bool const bSupportsShaderReplicated =
			shader_replicated_composites_features.shaderReplicatedComposites == vk::True;
		if (bSupportsExtensions && bSupportsQueues &&
			bSupportsCooperativeVector && bSupportsShaderReplicated) {
			return true;
		}
		return false;
	}

	inline float GetNsPerTick() const { return static_cast<float>(GetProperties10().limits.timestampPeriod); }

	vk::PhysicalDeviceCooperativeVectorFeaturesNV           cooperative_vector_features{};
	vk::PhysicalDeviceShaderReplicatedCompositesFeaturesEXT shader_replicated_composites_features{};
	vk::PhysicalDeviceCooperativeVectorPropertiesNV         cooperative_vector_properties{};

};