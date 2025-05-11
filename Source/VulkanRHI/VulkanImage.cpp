module VulkanRHI;
import :Image;
import :Enums;

import vulkan_hpp;
import vk_mem_alloc;
import VulkanModule;
import std;

namespace VulkanRHI {
Image::Image(vk::Image image, vk::ImageView view, vk::Extent3D const& extent)
	: vk::Image(image), view(view), from_swapchain(true) {
	info.image_info.extent        = extent;
	info.image_info.initialLayout = vk::ImageLayout::eUndefined;
	info.aspect                   = vk::ImageAspectFlagBits::eColor;
}

Image::Image(Image&& other) noexcept
	: vk::Image(std::exchange(static_cast<vk::Image&>(other), {})),
	  //   ResourceBase<vk::Device>(std::move(other)), view(std::exchange(other.view, {})),
	  allocation(std::exchange(other.allocation, {})),
	  device(std::move(other.device)),
	  info(std::move(other.info)),
	  from_swapchain(std::move(other.from_swapchain)) {}

Image& Image::operator=(Image&& other) noexcept {
	if (this != &other) {
		Destroy();
		vk::Image::operator=(std::exchange(static_cast<vk::Image&>(other), {}));
		device         = std::move(other.device);
		view           = std::exchange(other.view, {});
		allocation     = std::exchange(other.allocation, {});
		info           = std::move(other.info);
		from_swapchain = std::move(other.from_swapchain);
	}
	return *this;
}

Image::~Image() { Destroy(); }

auto Image::Create(vk::Device device, VmaAllocator vma_allocator, vk::AllocationCallbacks const* vk_allocator, ImageInfo const& info) -> vk::Result {

	this->device        = device;
	this->vma_allocator = vma_allocator;
	this->vk_allocator  = vk_allocator;

	this->info = info;

	return Recreate(info.image_info.extent);
}

auto Image::Recreate(vk::Extent3D const& extent) -> vk::Result {
	if (!IsValid()) return vk::Result::eErrorInitializationFailed;
	Destroy();
	this->info.image_info.extent = extent;

	VmaAllocationCreateInfo const allocInfo = {
		.usage          = VMA_MEMORY_USAGE_AUTO,
		.preferredFlags = VkMemoryPropertyFlags(
			GetUsage() & vk::ImageUsageFlagBits::eTransientAttachment ? VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT : 0),
	};

	vk::Result result;
	do {
		result =
			vk::Result(vmaCreateImage(vma_allocator, reinterpret_cast<VkImageCreateInfo const*>(&info.image_info),
									  &allocInfo,
									  reinterpret_cast<VkImage*>(static_cast<vk::Image*>(this)),
									  &allocation, nullptr));
		if (result != vk::Result::eSuccess) break;

		vk::ImageViewCreateInfo viewInfo{
			.image    = *this,
			.viewType = GetArrayLayers() == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::eCube,
			.format   = GetFormat(),
			.subresourceRange{.aspectMask     = GetAspect(),
							  .baseMipLevel   = 0,
							  .levelCount     = GetMipLevels(),
							  .baseArrayLayer = 0,
							  .layerCount     = GetArrayLayers()},
		};

		// Create image view 
		result = device.createImageView(&viewInfo, vk_allocator, &view);
		if (result != vk::Result::eSuccess) break;

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
