module;

export module SamplesCommon:VulkanUtils;

import vulkan_hpp;
import std;
import float16;

using numeric::float16_t;
 
export template <typename T>
constexpr inline auto GetVulkanComponentType() -> vk::ComponentTypeKHR {
	if constexpr (std::is_same_v<T, float16_t>) {
		return vk::ComponentTypeKHR::eFloat16;
	} else if constexpr (std::is_same_v<T, float>) {
		return vk::ComponentTypeKHR::eFloat32;
	} else if constexpr (std::is_same_v<T, std::int8_t>) {
		return vk::ComponentTypeKHR::eSint8;
	} else if constexpr (std::is_same_v<T, std::int16_t>) {
		return vk::ComponentTypeKHR::eSint16;
	} else if constexpr (std::is_same_v<T, std::int32_t>) {
		return vk::ComponentTypeKHR::eSint32;
	} else if constexpr (std::is_same_v<T, std::int64_t>) {
		return vk::ComponentTypeKHR::eSint64;
	} else if constexpr (std::is_same_v<T, std::uint8_t>) {
		return vk::ComponentTypeKHR::eUint8;
	} else if constexpr (std::is_same_v<T, std::uint16_t>) {
		return vk::ComponentTypeKHR::eUint16;
	} else if constexpr (std::is_same_v<T, std::uint32_t>) {
		return vk::ComponentTypeKHR::eUint32;
	} else if constexpr (std::is_same_v<T, std::uint64_t>) {
		return vk::ComponentTypeKHR::eUint64;
	}

	static_assert(false, "Unsupported type.");
}