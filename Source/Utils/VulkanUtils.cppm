export module NeuralGraphics:VulkanUtils;
import std;
import vulkan_hpp;

export
namespace ng::inline Utils {

template <typename T>
concept VulkanStructureT = requires(T t) { t.pNext; };

template <VulkanStructureT ExistingStructure, VulkanStructureT NewStructure>
inline void AddToPNext(ExistingStructure& existing, NewStructure& new_structure) {
	new_structure.pNext = existing.pNext;
	existing.pNext      = &new_structure;
}

inline bool SupportsExtension(std::span<vk::ExtensionProperties const> available_extensions,
							  std::string_view                         extension) {
	return std::any_of(available_extensions.begin(), available_extensions.end(),
					   [&extension](vk::ExtensionProperties const& available_extension) {
						   return available_extension.extensionName == extension;
					   });
};

inline bool SupportsExtensions(std::span<vk::ExtensionProperties const> available_extensions,
							   std::span<char const* const>             extensions) {
	return std::all_of(
		extensions.begin(), extensions.end(), [available_extensions](std::string_view extension) {
			return std::any_of(available_extensions.begin(), available_extensions.end(),
							   [&extension](vk::ExtensionProperties const& available_extension) {
								   return available_extension.extensionName == extension;
							   });
		});
}

constexpr auto GetVulkanComponentSize(vk::ComponentTypeKHR const type) -> std::size_t {
	switch (type) {
	case vk::ComponentTypeKHR::eFloat16:       return sizeof(std::uint16_t);
	case vk::ComponentTypeKHR::eFloat32:       return sizeof(float);
	case vk::ComponentTypeKHR::eFloat64:       return sizeof(double);
	case vk::ComponentTypeKHR::eSint8:         return sizeof(std::int8_t);
	case vk::ComponentTypeKHR::eSint16:        return sizeof(std::int16_t);
	case vk::ComponentTypeKHR::eSint32:        return sizeof(std::int32_t);
	case vk::ComponentTypeKHR::eSint64:        return sizeof(std::int64_t);
	case vk::ComponentTypeKHR::eUint8:         return sizeof(std::uint8_t);
	case vk::ComponentTypeKHR::eUint16:        return sizeof(std::uint16_t);
	case vk::ComponentTypeKHR::eUint32:        return sizeof(std::uint32_t);
	case vk::ComponentTypeKHR::eUint64:        return sizeof(std::uint64_t);
	case vk::ComponentTypeKHR::eSint8PackedNV: return sizeof(std::uint8_t);
	case vk::ComponentTypeKHR::eUint8PackedNV: return sizeof(std::uint8_t);
	case vk::ComponentTypeKHR::eFloatE4M3NV:   return sizeof(std::uint16_t);
	case vk::ComponentTypeKHR::eFloatE5M2NV:   return sizeof(std::uint16_t);
	default:
		return 0;
	}
}
} // namespace ng::inline Utils
