export module VulkanRHI:Image;

import std;
import vulkan_hpp;
import vk_mem_alloc;

export namespace VulkanRHI {

struct ImageInfo {
	vk::ImageCreateInfo        create_info;
	vk::ImageAspectFlags const aspect;
	std::string_view const     name = "";
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

	// Manually free resources, safe to call multiple times
	void Destroy();

	// Do not call
	inline void SetLayout(vk::ImageLayout const layout) { this->layout = layout; }

	inline bool IsFromSwapchain() const { return from_swapchain; }

	// Utility
	void SetDebugUtilsName(std::string_view const name);
	void SetDebugUtilsViewName(std::string_view const name);

	static auto MakeCreateInfo(vk::Format format, vk::Extent3D const& extent, vk::ImageUsageFlags usage) -> vk::ImageCreateInfo;

	inline auto GetDevice() const -> vk::Device { return device; }
	inline auto GetAllocator() const -> VmaAllocator { return vma_allocator; }
	inline auto GetAllocation() const -> VmaAllocation { return allocation; }
	inline auto GetAllocationInfo() const -> VmaAllocationInfo { return allocation_info; }

	inline auto GetLayout() const -> vk::ImageLayout { return layout; }
	inline auto GetExtent() const -> vk::Extent3D { return extent; }
	inline auto GetFormat() const -> vk::Format { return format; }
	inline auto GetUsage() const -> vk::ImageUsageFlags { return usage; }

	inline auto GetView() const -> vk::ImageView { return view; }
	inline auto GetAspect() const -> vk::ImageAspectFlags { return aspect; }

	bool IsValid() const { return vk::Image::operator bool(); }

private:
	vk::Device device;
	vk::AllocationCallbacks const* vk_allocator;

	VmaAllocator      vma_allocator;
	VmaAllocation     allocation;
	VmaAllocationInfo allocation_info;

	// From Create info
	vk::ImageLayout     layout;
	vk::Extent3D        extent;
	vk::Format          format;
	vk::ImageUsageFlags usage;

	vk::ImageView        view;
	vk::ImageAspectFlags aspect;

	bool from_swapchain = false;
};

} // namespace VulkanRHI
