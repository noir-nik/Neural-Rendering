module SamplesCommon;
import VulkanRHI;
import NeuralGraphics;
import vulkan_hpp;
import std;

bool PhysicalDevice::IsSuitable(vk::SurfaceKHR const& surface, std::span<char const* const> extensions) {
	bool const bSupportsExtensions =
		SupportsExtensions(extensions);
	bool const bSupportsQueues =
		SupportsQueue({.flags = vk::QueueFlagBits::eGraphics, .surface = surface});
	bool const bSupportsCooperativeVector =
		cooperative_vector_features.cooperativeVector == vk::True;
	bool const bSupportsShaderReplicated =
		shader_replicated_composites_features.shaderReplicatedComposites == vk::True;
	return true;
	if (bSupportsExtensions && bSupportsQueues && bSupportsCooperativeVector && bSupportsShaderReplicated) {
	}
	return false;
}
