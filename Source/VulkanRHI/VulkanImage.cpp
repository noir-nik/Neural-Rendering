module VulkanRHI;
import :Image;
import :Enums;

import vulkan_hpp;
import vk_mem_alloc;
import VulkanModule;
import std;

namespace VulkanRHI {
Image::Image(vk::Image image, vk::ImageView view, vk::Extent3D const& extent)
	: vk::Image(image),
	  view(view),
	  layout(vk::ImageLayout::eUndefined),
	  aspect(vk::ImageAspectFlagBits::eColor),
	  from_swapchain(true),
	  extent(extent) {}

Image::Image(Image&& other) noexcept
	: vk::Image(std::exchange(static_cast<vk::Image&>(other), {})),
	  //   ResourceBase<vk::Device>(std::move(other)), view(std::exchange(other.view, {})),
	  allocation(std::exchange(other.allocation, {})),
	  layout(std::move(other.layout)),
	  device(std::move(other.device)),
	  aspect(std::move(other.aspect)),
	  extent(std::move(other.extent)),
	  format(std::move(other.format)),
	  usage(std::move(other.usage)),
	  from_swapchain(std::move(other.from_swapchain)) {}

Image& Image::operator=(Image&& other) noexcept {
	if (this != &other) {
		Destroy();
		vk::Image::operator=(std::exchange(static_cast<vk::Image&>(other), {}));
		device         = std::move(other.device);
		view           = std::exchange(other.view, {});
		allocation     = std::exchange(other.allocation, {});
		layout         = std::move(other.layout);
		aspect         = std::move(other.aspect);
		extent         = std::move(other.extent);
		format         = std::move(other.format);
		usage          = std::move(other.usage);
		from_swapchain = std::move(other.from_swapchain);
	}
	return *this;
}

Image::~Image() { Destroy(); }

auto Image::Create(vk::Device device, VmaAllocator vma_allocator, vk::AllocationCallbacks const* vk_allocator, ImageInfo const& info) -> vk::Result {

	this->device        = device;
	this->vma_allocator = vma_allocator;
	this->vk_allocator  = vk_allocator;

	this->extent = info.create_info.extent;
	this->format = info.create_info.format;
	this->usage  = info.create_info.usage;
	this->layout = info.create_info.initialLayout;
	this->aspect = info.aspect;

	VmaAllocationCreateInfo const allocInfo = {
		.usage          = VMA_MEMORY_USAGE_AUTO,
		.preferredFlags = VkMemoryPropertyFlags(
			info.create_info.usage & vk::ImageUsageFlagBits::eTransientAttachment ? VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT : 0),
	};

	vk::Result result;
	do {
		result =
			vk::Result(vmaCreateImage(vma_allocator, reinterpret_cast<VkImageCreateInfo const*>(&info),
									  &allocInfo,
									  reinterpret_cast<VkImage*>(static_cast<vk::Image*>(this)),
									  &allocation, nullptr));
		if (result != vk::Result::eSuccess) continue;

		vk::ImageViewCreateInfo viewInfo{
			.image    = *this,
			.viewType = info.create_info.arrayLayers == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::eCube,
			.format   = info.create_info.format,
			.subresourceRange{.aspectMask     = aspect,
							  .baseMipLevel   = 0,
							  .levelCount     = info.create_info.mipLevels,
							  .baseArrayLayer = 0,
							  .layerCount     = info.create_info.arrayLayers}};

		// todo(nm): Create image view only if usage if Sampled or Storage or other fitting
		result = device.createImageView(&viewInfo, vk_allocator, &view);
		if (result != vk::Result::eSuccess) continue;

		return result;
	} while (false);

	Destroy();
	return result;
}

void Image::Destroy() {
	if (!IsValid()) return;
	if (!from_swapchain) {
		device.destroyImageView(view, vk_allocator);
		vmaDestroyImage(vma_allocator, *this, allocation);
		vk::Image::operator=(nullptr);
		view = vk::ImageView{};
	}
}
} // namespace VulkanRHI
