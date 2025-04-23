export module VulkanRHI:Swapchain;

import vulkan_hpp;
import std;

export namespace VulkanRHI {

using u32 = std::uint32_t;

struct SwapchainInfo {
	// Necessary data
	vk::SurfaceKHR surface            = nullptr;
	vk::Extent2D   extent             = {};
	u32            queue_family_index = ~0; // for command pool
	// Optional
	u32 frames_in_flight  = 3;
	u32 additional_images = 0;
	// Preferred color format, not guaranteed, get actual format after creation with GetFormat()
	vk::Format         preferred_format = vk::Format::eR8G8B8A8Unorm;
	vk::ColorSpaceKHR  color_space      = vk::ColorSpaceKHR::eSrgbNonlinear;
	vk::PresentModeKHR present_mode     = vk::PresentModeKHR::eMailbox;
};

struct SwapchainFrameData {
public:
	auto GetCommandPool() -> vk::CommandPool& { return command_pool; }
	auto GetCommandPool() const -> vk::CommandPool const& { return command_pool; }
	auto GetCommandBuffer() -> vk::CommandBuffer& { return command_buffer; }
	auto GetCommandBuffer() const -> vk::CommandBuffer const& { return command_buffer; }
	auto GetFence() -> vk::Fence& { return fence; }
	auto GetFence() const -> vk::Fence const& { return fence; }
	auto GetImageAvailableSemaphore() -> vk::Semaphore& { return image_available_semaphore; }
	auto GetImageAvailableSemaphore() const -> vk::Semaphore const& { return image_available_semaphore; }
	auto GetRenderFinishedSemaphore() -> vk::Semaphore& { return render_finished_semaphore; }
	auto GetRenderFinishedSemaphore() const -> vk::Semaphore const& { return render_finished_semaphore; }

private:
	vk::CommandPool   command_pool{};
	vk::CommandBuffer command_buffer{};
	vk::Fence         fence{};
	vk::Semaphore     image_available_semaphore{};
	vk::Semaphore     render_finished_semaphore{};
};

class Swapchain : public vk::SwapchainKHR {
public:
	Swapchain() = default;

	Swapchain(Swapchain const&)            = delete;
	Swapchain& operator=(Swapchain const&) = delete;
	Swapchain(Swapchain&& other) noexcept;
	Swapchain& operator=(Swapchain&& other) noexcept;

	~Swapchain();

	[[nodiscard]] auto Create(vk::Device                     device,
							  vk::PhysicalDevice             physical_device,
							  SwapchainInfo const&           info,
							  vk::AllocationCallbacks const* allocator = nullptr) -> vk::Result;

	[[nodiscard]] auto Recreate(u32 width, u32 height) -> vk::Result;

	void Destroy();

	[[nodiscard]] auto AcquireNextImage() -> vk::Result;
	[[nodiscard]] auto SubmitAndPresent(vk::Queue submit, vk::Queue present) -> vk::Result;

	void EndFrame();

	auto GetFrameData() -> std::span<SwapchainFrameData> { return frames; }
	auto GetFrameData() const -> std::span<SwapchainFrameData const> { return frames; }
	auto GetCurrentFrameData() -> SwapchainFrameData& { return frames[current_frame_index]; }
	auto GetCurrentFrameData() const -> SwapchainFrameData const& { return frames[current_frame_index]; }

	auto GetDevice() -> vk::Device& { return device; }
	auto GetDevice() const -> vk::Device const& { return device; }
	auto GetPhysicalDevice() -> vk::PhysicalDevice& { return physical_device; }
	auto GetPhysicalDevice() const -> vk::PhysicalDevice const& { return physical_device; }
	auto GetAllocator() const -> vk::AllocationCallbacks const* { return allocator; }

	auto GetCurrentCommandPool() -> vk::CommandPool& { return frames[current_frame_index].GetCommandPool(); }
	auto GetCurrentCommandPool() const -> vk::CommandPool const& { return frames[current_frame_index].GetCommandPool(); }
	auto GetCurrentCommandBuffer() -> vk::CommandBuffer& { return frames[current_frame_index].GetCommandBuffer(); }
	auto GetCurrentCommandBuffer() const -> vk::CommandBuffer const& { return frames[current_frame_index].GetCommandBuffer(); }
	auto GetCurrentFence() -> vk::Fence& { return frames[current_frame_index].GetFence(); }
	auto GetCurrentFence() const -> vk::Fence const& { return frames[current_frame_index].GetFence(); }
	auto GetCurrentImageAvailableSemaphore() -> vk::Semaphore& { return frames[current_frame_index].GetImageAvailableSemaphore(); }
	auto GetCurrentImageAvailableSemaphore() const -> vk::Semaphore const& { return frames[current_frame_index].GetImageAvailableSemaphore(); }
	auto GetCurrentRenderFinishedSemaphore() -> vk::Semaphore& { return frames[current_frame_index].GetRenderFinishedSemaphore(); }
	auto GetCurrentRenderFinishedSemaphore() const -> vk::Semaphore const& { return frames[current_frame_index].GetRenderFinishedSemaphore(); }

	auto GetCurrentImage() -> vk::Image& { return images[current_image_index]; }
	auto GetCurrentImage() const -> vk::Image const& { return images[current_image_index]; }
	auto GetCurrentImageView() -> vk::ImageView& { return image_views[current_image_index]; }
	auto GetCurrentImageView() const -> vk::ImageView const& { return image_views[current_image_index]; }
	auto GetCurrentFrameIndex() -> u32& { return current_frame_index; }
	auto GetCurrentFrameIndex() const -> u32 const& { return current_frame_index; }
	auto GetCurrentImageIndex() -> u32& { return current_image_index; }
	auto GetCurrentImageIndex() const -> u32 const& { return current_image_index; }

	auto GetSurface() -> vk::SurfaceKHR& { return info.surface; }
	auto GetSurface() const -> vk::SurfaceKHR const& { return info.surface; }
	auto GetExtent() -> vk::Extent2D& { return info.extent; }
	auto GetExtent() const -> vk::Extent2D const& { return info.extent; }
	auto GetQueueFamilyIndex() -> u32& { return info.queue_family_index; }
	auto GetQueueFamilyIndex() const -> u32 const& { return info.queue_family_index; }
	auto GetFramesInFlight() -> u32& { return info.frames_in_flight; }
	auto GetFramesInFlight() const -> u32 const& { return info.frames_in_flight; }
	auto GetAdditionalImages() -> u32& { return info.additional_images; }
	auto GetAdditionalImages() const -> u32 const& { return info.additional_images; }
	auto GetFormat() -> vk::Format& { return info.preferred_format; }
	auto GetFormat() const -> vk::Format const& { return info.preferred_format; }
	auto GetColorSpace() -> vk::ColorSpaceKHR& { return info.color_space; }
	auto GetColorSpace() const -> vk::ColorSpaceKHR const& { return info.color_space; }
	auto GetPresentMode() -> vk::PresentModeKHR& { return info.present_mode; }
	auto GetPresentMode() const -> vk::PresentModeKHR const& { return info.present_mode; }

private:
	bool SupportsFormat(vk::Format format, vk::ImageTiling tiling, vk::FormatFeatureFlags features);
	auto ChooseSurfaceFormat(vk::SurfaceFormatKHR const& desired_surface_format) -> vk::SurfaceFormatKHR;
	auto ChoosePresentMode(vk::PresentModeKHR const& desired_present_mode) -> vk::PresentModeKHR;
	auto ChooseExtent(vk::Extent2D const& desired_extent) -> vk::Extent2D;

	[[nodiscard]] auto CreateSurfaceFormats() -> vk::Result;
	[[nodiscard]] auto CreateSwapchain() -> vk::Result;
	[[nodiscard]] auto CreateImages() -> vk::Result;
	[[nodiscard]] auto CreateFrames() -> vk::Result;

	vk::Device         device;
	vk::PhysicalDevice physical_device;

	vk::AllocationCallbacks const* allocator = nullptr;

	vk::SurfaceCapabilitiesKHR surface_capabilities;

	std::vector<vk::PresentModeKHR>   available_present_modes;
	std::vector<vk::SurfaceFormatKHR> available_surface_formats;

	std::vector<SwapchainFrameData> frames;
	std::vector<vk::Image>          images;
	std::vector<vk::ImageView>      image_views;
	u32                             current_frame_index = 0;
	u32                             current_image_index = 0;
	SwapchainInfo                   info;
};
} // namespace VulkanRHI
