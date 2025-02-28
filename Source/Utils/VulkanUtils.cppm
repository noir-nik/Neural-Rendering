export module NeuralGraphics:VulkanUtils;
import std;
import vulkan_hpp;

export namespace ng {
inline namespace Utils {

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

} // namespace Utils
} // namespace ng
