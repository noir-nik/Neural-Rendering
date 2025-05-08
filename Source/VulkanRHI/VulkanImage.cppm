export module VulkanRHI:Image;

import std;
import vulkan_hpp;
import vk_mem_alloc;

export namespace VulkanRHI {

struct ImageInfo {
	vk::ImageCreateInfo  image_info;
	vk::ImageAspectFlags aspect;
	std::uint32_t        mip_levels   = 1;
	std::uint32_t        array_layers = 1;
};

class Image : public vk::Image {
public:
	Image() = default;

	// From swapchain
	Image(vk::Image image, vk::ImageView view, vk::Extent3D const& extent);

	Image(Image const&)            = delete;
	Image& operator=(Image const&) = delete;
	Image(Image&& other) noexcept;
	Image& operator=(Image&& other) noexcept;

	// Calls Destroy
	~Image();

	auto Create(vk::Device device, VmaAllocator vma_allocator, vk::AllocationCallbacks const* vk_allocator, ImageInfo const& info) -> vk::Result;

	auto Recreate(vk::Extent3D const& extent) -> vk::Result;

	// Manually free resources, safe to call multiple times
	void Destroy();

	// Do not call
	inline void SetLayout(vk::ImageLayout const layout) { this->info.image_info.initialLayout = layout; }

	inline bool IsFromSwapchain() const { return from_swapchain; }

	// Utility
	void SetDebugUtilsName(std::string_view const name);
	void SetDebugUtilsViewName(std::string_view const name);

	static auto MakeCreateInfo(vk::Format format, vk::Extent3D const& extent, vk::ImageUsageFlags usage) -> vk::ImageCreateInfo;

	inline auto GetDevice() const -> vk::Device { return device; }
	inline auto GetAllocator() const -> VmaAllocator { return vma_allocator; }
	inline auto GetAllocation() const -> VmaAllocation { return allocation; }
	inline auto GetAllocationInfo() const -> VmaAllocationInfo { return allocation_info; }

	inline auto GetLayout() const -> vk::ImageLayout { return info.image_info.initialLayout; }
	inline auto GetExtent() const -> vk::Extent3D { return info.image_info.extent; }
	inline auto GetFormat() const -> vk::Format { return info.image_info.format; }
	inline auto GetUsage() const -> vk::ImageUsageFlags { return info.image_info.usage; }

	inline auto GetView() const -> vk::ImageView { return view; }
	inline auto GetAspect() const -> vk::ImageAspectFlags { return info.aspect; }
	inline auto GetMipLevels() const -> std::uint32_t { return info.mip_levels; }
	inline auto GetArrayLayers() const -> std::uint32_t { return info.array_layers; }

	bool IsValid() const { return vk::Image::operator bool(); }

private:
	vk::Device                     device;
	vk::AllocationCallbacks const* vk_allocator;

	VmaAllocator      vma_allocator;
	VmaAllocation     allocation;
	VmaAllocationInfo allocation_info;

	vk::ImageView view;

	ImageInfo info;

	bool from_swapchain = false;
};

} // namespace VulkanRHI
