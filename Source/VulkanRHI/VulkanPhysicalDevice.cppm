export module VulkanRHI:PhysicalDevice;
import vulkan_hpp;
import std;

export namespace VulkanRHI {
using u32 = std::uint32_t;

struct QueueFamilyInfo {
	vk::QueueFlags flags;
	vk::QueueFlags undesired_flags = {};
	vk::SurfaceKHR surface         = {};
};

class PhysicalDevice : public vk::PhysicalDevice {
public:
	struct Features {
		Features() {
			features2.pNext = &vulkan11;
			vulkan11.pNext  = &vulkan12;
			vulkan12.pNext  = &vulkan13;
		}
		vk::PhysicalDeviceFeatures2        features2;
		vk::PhysicalDeviceVulkan11Features vulkan11;
		vk::PhysicalDeviceVulkan12Features vulkan12;
		vk::PhysicalDeviceVulkan13Features vulkan13;
	};

	struct Properties {
		Properties() {
			properties2.pNext = &vulkan11;
			vulkan11.pNext    = &vulkan12;
			vulkan12.pNext    = &vulkan13;
		}
		vk::PhysicalDeviceProperties2        properties2;
		vk::PhysicalDeviceVulkan11Properties vulkan11;
		vk::PhysicalDeviceVulkan12Properties vulkan12;
		vk::PhysicalDeviceVulkan13Properties vulkan13;
	};

	inline void        Assign(vk::PhysicalDevice const& physical_device) { vk::PhysicalDevice::operator=(physical_device); }
	[[nodiscard]] auto GetDetails() -> vk::Result;

	bool SupportsExtension(std::string_view extension) const;
	bool SupportsExtensions(std::span<char const* const> extensions) const;
	bool SupportsQueue(QueueFamilyInfo const& info) const;
	bool SupportsQueues(std::span<QueueFamilyInfo const> infos) const;

	auto GetQueueFamilyIndex(vk::QueueFlags desired_flags, vk::QueueFlags undesired_flags = {}) const -> std::optional<u32>;
	auto GetQueueFamilyIndex(QueueFamilyInfo const& info) const -> std::pair<vk::Result, std::optional<u32>>;

	auto GetDedicatedTransferQueueFamilyIndex() const -> std::optional<u32>;
	auto GetDedicatedComputeQueueFamilyIndex() const -> std::optional<u32>;
	auto GetQueueCount(u32 queue_family_index) const -> u32;
	auto GetQueueFamilyProperties(u32 queue_family_index) const -> vk::QueueFamilyProperties const&;
	auto GetMaxPushConstantsSize() const -> u32;

	constexpr inline auto GetFeatures2() -> vk::PhysicalDeviceFeatures2& { return features.features2; }
	constexpr inline auto GetFeatures10() -> vk::PhysicalDeviceFeatures& { return features.features2.features; }
	constexpr inline auto GetFeatures11() -> vk::PhysicalDeviceVulkan11Features& { return features.vulkan11; }
	constexpr inline auto GetFeatures12() -> vk::PhysicalDeviceVulkan12Features& { return features.vulkan12; }
	constexpr inline auto GetFeatures13() -> vk::PhysicalDeviceVulkan13Features& { return features.vulkan13; }

	constexpr inline auto GetFeatures2() const -> vk::PhysicalDeviceFeatures2 const& { return features.features2; }
	constexpr inline auto GetFeatures10() const -> vk::PhysicalDeviceFeatures const& { return features.features2.features; }
	constexpr inline auto GetFeatures11() const -> vk::PhysicalDeviceVulkan11Features const& { return features.vulkan11; }
	constexpr inline auto GetFeatures12() const -> vk::PhysicalDeviceVulkan12Features const& { return features.vulkan12; }
	constexpr inline auto GetFeatures13() const -> vk::PhysicalDeviceVulkan13Features const& { return features.vulkan13; }

	constexpr inline auto GetProperties2() -> vk::PhysicalDeviceProperties2& { return properties.properties2; }
	constexpr inline auto GetProperties10() -> vk::PhysicalDeviceProperties& { return properties.properties2.properties; }
	constexpr inline auto GetProperties11() -> vk::PhysicalDeviceVulkan11Properties& { return properties.vulkan11; }
	constexpr inline auto GetProperties12() -> vk::PhysicalDeviceVulkan12Properties& { return properties.vulkan12; }
	constexpr inline auto GetProperties13() -> vk::PhysicalDeviceVulkan13Properties& { return properties.vulkan13; }

	constexpr inline auto GetProperties2() const -> vk::PhysicalDeviceProperties2 const& { return properties.properties2; }
	constexpr inline auto GetProperties10() const -> vk::PhysicalDeviceProperties const& { return properties.properties2.properties; }
	constexpr inline auto GetProperties11() const -> vk::PhysicalDeviceVulkan11Properties const& { return properties.vulkan11; }
	constexpr inline auto GetProperties12() const -> vk::PhysicalDeviceVulkan12Properties const& { return properties.vulkan12; }
	constexpr inline auto GetProperties13() const -> vk::PhysicalDeviceVulkan13Properties const& { return properties.vulkan13; }

	constexpr inline auto GetMemoryProperties() const -> vk::PhysicalDeviceMemoryProperties2 const& { return memory_properties; }
	constexpr inline auto GetQueueFamilyProperties() const -> std::vector<vk::QueueFamilyProperties2> const& { return queue_family_properties; }
	constexpr inline auto GetAvailableExtensions() const -> std::vector<vk::ExtensionProperties> const& { return available_extensions; }
	constexpr inline auto GetMaxSamples() const -> vk::SampleCountFlagBits { return max_samples; }

private:
	Features features;

	Properties properties;

	// Memory properties
	vk::PhysicalDeviceMemoryProperties2 memory_properties;

	// Queue properties
	std::vector<vk::QueueFamilyProperties2> queue_family_properties;

	// Extensions
	std::vector<vk::ExtensionProperties> available_extensions;

	vk::SampleCountFlagBits max_samples = vk::SampleCountFlagBits::e1;
};

} // namespace VulkanRHI
