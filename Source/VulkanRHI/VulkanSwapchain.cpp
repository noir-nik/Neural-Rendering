module VulkanRHI;
import :Swapchain;

import vulkan_hpp;
import std;

using u64 = std::uint64_t;

#define RETURN_ON_ERROR(func) \
	{ \
		vk::Result local_result_ = (func); \
		if (local_result_ != vk::Result::eSuccess) { \
			return local_result_; \
		} \
	}

namespace VulkanRHI {

Swapchain::~Swapchain() { (void)Destroy(); }

Swapchain::Swapchain(Swapchain&& other) noexcept {
	(void)Destroy(); // Destroy existing resources
	this->operator=(std::move(other));
}

Swapchain& Swapchain::operator=(Swapchain&& other) noexcept {
	if (this != &other) {
		(void)Destroy(); // Destroy existing resources
		vk::SwapchainKHR::operator=(std::exchange(static_cast<vk::SwapchainKHR&>(other), {}));
		info                      = std::move(other.info);
		images                    = std::move(other.images);
		image_views               = std::exchange(other.image_views, {});
		frames                    = std::exchange(other.frames, {});
		available_present_modes   = std::move(other.available_present_modes);
		available_surface_formats = std::move(other.available_surface_formats);
		current_frame_index       = other.current_frame_index;
		current_image_index       = other.current_image_index;
	}
	return *this;
}

auto Swapchain::Create(vk::Device                     device,
					   vk::PhysicalDevice             physical_device,
					   SwapchainInfo const&           init_info,
					   vk::AllocationCallbacks const* allocator) -> vk::Result {
	this->device          = device;
	this->physical_device = physical_device;
	this->info            = init_info;
	this->allocator       = allocator;

	RETURN_ON_ERROR(CreateSurfaceFormats());
	auto [format, colorSpace] = ChooseSurfaceFormat({info.preferred_format,
													 info.color_space});
	info.preferred_format     = format;
	info.color_space          = colorSpace;
	info.present_mode         = ChoosePresentMode(info.present_mode);
	info.extent               = ChooseExtent(info.extent);
	RETURN_ON_ERROR(CreateSwapchain());
	RETURN_ON_ERROR(CreateImages());
	RETURN_ON_ERROR(CreateFrames());

	current_frame_index = 0;
	current_image_index = 0;
	return vk::Result::eSuccess;
}

auto Swapchain::Recreate(u32 width, u32 height) -> vk::Result {
	if (width <= 0 || height <= 0) {
		return vk::Result::eErrorUnknown;
	}

	for (auto& image_view : image_views) {
		GetDevice().destroyImageView(image_view, GetAllocator());
	}
	image_views.clear();

	RETURN_ON_ERROR(CreateSurfaceFormats());
	if (available_present_modes.empty() || available_surface_formats.empty()) [[unlikely]] {
		return vk::Result::eErrorUnknown;
	}
	info.extent               = ChooseExtent({width, height});
	auto [format, colorSpace] = ChooseSurfaceFormat({info.preferred_format,
													 info.color_space});
	info.preferred_format     = format;
	info.color_space          = colorSpace;
	info.present_mode         = ChoosePresentMode(info.present_mode);
	RETURN_ON_ERROR(CreateSwapchain());
	RETURN_ON_ERROR(CreateImages());

	return vk::Result::eSuccess;
}

void Swapchain::Destroy() {
	if (!GetDevice()) {
		return;
	}
	for (auto& frame : frames) {
		GetDevice().destroyCommandPool(frame.GetCommandPool(), GetAllocator());
		GetDevice().destroyFence(frame.GetFence(), GetAllocator());
		GetDevice().destroySemaphore(frame.GetImageAvailableSemaphore(), GetAllocator());
		GetDevice().destroySemaphore(frame.GetRenderFinishedSemaphore(), GetAllocator());
	}
	frames.clear();

	for (auto& image_view : image_views) {
		GetDevice().destroyImageView(image_view, GetAllocator());
	}
	images.clear();
	image_views.clear();

	available_present_modes.clear();
	available_surface_formats.clear();

	GetDevice().destroySwapchainKHR(*this, GetAllocator());

	vk::SwapchainKHR::operator=(vk::SwapchainKHR{});
	device = vk::Device{};
}

auto Swapchain::AcquireNextImage() -> vk::Result {
	vk::AcquireNextImageInfoKHR acquire_info{
		.swapchain  = *this,
		.timeout    = std::numeric_limits<u64>::max(),
		.semaphore  = GetCurrentImageAvailableSemaphore(),
		.fence      = nullptr,
		.deviceMask = 1,
	};
	return GetDevice().acquireNextImage2KHR(&acquire_info, &current_image_index);
}

// vkQueueSubmit2 + vkQueuePresentKHR
auto Swapchain::SubmitAndPresent(vk::Queue submit, vk::Queue present) -> vk::Result {

	vk::SemaphoreSubmitInfo     wait{.semaphore = GetCurrentImageAvailableSemaphore()};
	vk::SemaphoreSubmitInfo     signal{.semaphore = GetCurrentRenderFinishedSemaphore()};
	vk::CommandBufferSubmitInfo bufferSubmitInfo{.commandBuffer = GetCurrentCommandBuffer()};

	vk::SubmitInfo2 submit_info{
		.waitSemaphoreInfoCount   = 1,
		.pWaitSemaphoreInfos      = &wait,
		.commandBufferInfoCount   = 1,
		.pCommandBufferInfos      = &bufferSubmitInfo,
		.signalSemaphoreInfoCount = 1,
		.pSignalSemaphoreInfos    = &signal,
	};

	RETURN_ON_ERROR(submit.submit2(1, &submit_info, GetCurrentFence()));

	vk::Semaphore present_wait = GetCurrentRenderFinishedSemaphore();

	vk::PresentInfoKHR present_info{
		.waitSemaphoreCount = 1,
		.pWaitSemaphores    = &present_wait,
		.swapchainCount     = 1,
		.pSwapchains        = this,
		.pImageIndices      = &current_image_index,
		.pResults           = nullptr,
	};

	RETURN_ON_ERROR(present.presentKHR(&present_info));

	return vk::Result::eSuccess;
}

void Swapchain::EndFrame() {
	current_frame_index = (current_frame_index + 1) % info.frames_in_flight;
}

bool Swapchain::SupportsFormat(vk::Format format, vk::ImageTiling tiling, vk::FormatFeatureFlags features) {
	vk::FormatProperties props;
	GetPhysicalDevice().getFormatProperties(format, &props);

	if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
		return true;
	} else if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
		return true;
	}

	return false;
}

auto Swapchain::ChoosePresentMode(vk::PresentModeKHR const& desired_present_mode) -> vk::PresentModeKHR {
	for (auto const& mode : available_present_modes) {
		if (mode == static_cast<vk::PresentModeKHR>(desired_present_mode)) {
			return desired_present_mode;
		}
	}
	// FIFO is guaranteed to be available
	return vk::PresentModeKHR::eFifo;
}

auto Swapchain::ChooseExtent(vk::Extent2D const& desired_extent) -> vk::Extent2D {
	if (this->surface_capabilities.currentExtent.width != std::numeric_limits<u32>::max()) {
		return {
			surface_capabilities.currentExtent.width,
			surface_capabilities.currentExtent.height,
		};
	}
	return vk::Extent2D{
		.width = std::max(
			this->surface_capabilities.minImageExtent.width,
			std::min(this->surface_capabilities.maxImageExtent.width, desired_extent.width)),
		.height = std::max(
			this->surface_capabilities.minImageExtent.height,
			std::min(this->surface_capabilities.maxImageExtent.height, desired_extent.height))};
}

auto Swapchain::CreateSurfaceFormats() -> vk::Result {
	// get capabilities
	RETURN_ON_ERROR(GetPhysicalDevice().getSurfaceCapabilitiesKHR(info.surface, &surface_capabilities));

	// get surface formats
	u32 formatCount;
	RETURN_ON_ERROR(GetPhysicalDevice().getSurfaceFormatsKHR(info.surface, &formatCount, nullptr));
	if (formatCount != 0) {
		available_surface_formats.clear();
		available_surface_formats.resize(formatCount);
		RETURN_ON_ERROR(GetPhysicalDevice().getSurfaceFormatsKHR(info.surface, &formatCount, available_surface_formats.data()));
	}

	// get present modes
	u32 modeCount;
	RETURN_ON_ERROR(GetPhysicalDevice().getSurfacePresentModesKHR(info.surface, &modeCount, nullptr));
	if (modeCount != 0) {
		available_present_modes.clear();
		available_present_modes.resize(modeCount);
		RETURN_ON_ERROR(GetPhysicalDevice().getSurfacePresentModesKHR(info.surface, &modeCount, available_present_modes.data()));
	}
	return vk::Result::eSuccess;
}

auto Swapchain::ChooseSurfaceFormat(vk::SurfaceFormatKHR const& desired_surface_format) -> vk::SurfaceFormatKHR {
	for (auto const& availableFormat : available_surface_formats) {
		if (availableFormat.format == desired_surface_format.format &&
			availableFormat.colorSpace == desired_surface_format.colorSpace) {
			return {availableFormat.format, availableFormat.colorSpace};
		}
	}
	return {available_surface_formats[0].format, available_surface_formats[0].colorSpace};
}

auto Swapchain::CreateSwapchain() -> vk::Result {
	vk::SurfaceCapabilitiesKHR const& capabilities = surface_capabilities;

	// acquire additional images to prevent waiting for driver's internal operations
	u32 image_count = info.frames_in_flight + info.additional_images;

	if (image_count < capabilities.minImageCount) {
		image_count = capabilities.minImageCount;
	}

	// prevent exceeding the max image count
	if (capabilities.maxImageCount > 0 && image_count > capabilities.maxImageCount) {
		image_count = capabilities.maxImageCount;
	}

	// Create swapchain
	vk::SwapchainCreateInfoKHR createInfo{
		.surface          = info.surface,
		.minImageCount    = image_count,
		.imageFormat      = info.preferred_format,
		.imageColorSpace  = info.color_space,
		.imageExtent      = {info.extent.width, info.extent.height},
		.imageArrayLayers = 1,
		.imageUsage       = vk::ImageUsageFlagBits::eColorAttachment // if we want to render to a separate image first to perform post-processing
					  | vk::ImageUsageFlagBits::eTransferDst,        // we should add this image usage
		.imageSharingMode      = vk::SharingMode::eExclusive,        // don't support different graphics and present family
		.queueFamilyIndexCount = 0,                                  // only when imageSharingMode is VK_SHARING_MODE_CONCURRENT
		.pQueueFamilyIndices   = nullptr,                            // only when imageSharingMode is VK_SHARING_MODE_CONCURRENT
		.preTransform          = capabilities.currentTransform,      // blend the images with other windows in the window system
		.compositeAlpha        = vk::CompositeAlphaFlagBitsKHR::eOpaque,
		.presentMode           = info.present_mode,
		.clipped               = true, // clip pixels behind our window
		.oldSwapchain          = *this,
	};

	RETURN_ON_ERROR(GetDevice().createSwapchainKHR(&createInfo, GetAllocator(), this));

	GetDevice().destroySwapchainKHR(createInfo.oldSwapchain, GetAllocator());
	return vk::Result::eSuccess;
}

auto Swapchain::CreateImages() -> vk::Result {
	u32        imageCount;
	vk::Result result;
	RETURN_ON_ERROR(GetDevice().getSwapchainImagesKHR(*this, &imageCount, nullptr));
	images.resize(imageCount);
	image_views.resize(imageCount);
	RETURN_ON_ERROR(GetDevice().getSwapchainImagesKHR(*this, &imageCount, images.data()));

	// Create image view
	for (u32 i = 0; i < imageCount; ++i) {
		vk::ImageViewCreateInfo viewInfo{
			.image    = images[i],
			.viewType = vk::ImageViewType::e2D,
			.format   = info.preferred_format,
			.subresourceRange{
				.aspectMask     = vk::ImageAspectFlagBits::eColor,
				.baseMipLevel   = 0,
				.levelCount     = 1,
				.baseArrayLayer = 0,
				.layerCount     = 1,
			},
		};
		RETURN_ON_ERROR(GetDevice().createImageView(&viewInfo, GetAllocator(), &image_views[i]));
	}
	return vk::Result::eSuccess;
}

auto Swapchain::CreateFrames() -> vk::Result {
	frames.resize(info.frames_in_flight);
	for (u32 i = 0; i < info.frames_in_flight; ++i) {
		vk::CommandPoolCreateInfo pool_info{.queueFamilyIndex = info.queue_family_index};
		RETURN_ON_ERROR(GetDevice().createCommandPool(&pool_info, GetAllocator(), &frames[i].GetCommandPool()));

		vk::CommandBufferAllocateInfo alloc_info{
			.commandPool        = frames[i].GetCommandPool(),
			.level              = vk::CommandBufferLevel::ePrimary,
			.commandBufferCount = 1,
		};
		RETURN_ON_ERROR(GetDevice().allocateCommandBuffers(&alloc_info, &frames[i].GetCommandBuffer()));

		vk::FenceCreateInfo fence_info{.flags = vk::FenceCreateFlagBits::eSignaled};
		RETURN_ON_ERROR(GetDevice().createFence(&fence_info, GetAllocator(), &frames[i].GetFence()));

		vk::SemaphoreCreateInfo semaphoreInfo{};
		RETURN_ON_ERROR(GetDevice().createSemaphore(&semaphoreInfo, GetAllocator(), &frames[i].GetImageAvailableSemaphore()));
		RETURN_ON_ERROR(GetDevice().createSemaphore(&semaphoreInfo, GetAllocator(), &frames[i].GetRenderFinishedSemaphore()));
	}

	return vk::Result::eSuccess;
}

} // namespace VulkanRHI
