export module VulkanRHI:CommandBuffer;

import vulkan_hpp;
import std;

using u32 = std::uint32_t;

export namespace VulkanRHI {

/* ===== Synchronization 2 Barriers ===== */
struct BufferBarrier {
	vk::Buffer              buffer;
	vk::PipelineStageFlags2 srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	vk::AccessFlags2        srcAccessMask       = vk::AccessFlagBits2::eShaderWrite;
	vk::PipelineStageFlags2 dstStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	vk::AccessFlags2        dstAccessMask       = vk::AccessFlagBits2::eShaderRead;
	u32                     srcQueueFamilyIndex = vk::QueueFamilyIgnored;
	u32                     dstQueueFamilyIndex = vk::QueueFamilyIgnored;
	vk::DeviceSize          offset              = 0;
	vk::DeviceSize          size                = vk::WholeSize;
};

struct ImageBarrier {
	vk::Image               image;
	vk::ImageAspectFlags    aspectMask          = vk::ImageAspectFlagBits::eColor;
	vk::ImageLayout         oldLayout           = vk::ImageLayout::eUndefined;
	vk::ImageLayout         newLayout           = vk::ImageLayout::eUndefined;
	vk::PipelineStageFlags2 srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	vk::AccessFlags2        srcAccessMask       = vk::AccessFlagBits2::eShaderWrite;
	vk::PipelineStageFlags2 dstStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	vk::AccessFlags2        dstAccessMask       = vk::AccessFlagBits2::eShaderRead;
	u32                     srcQueueFamilyIndex = vk::QueueFamilyIgnored;
	u32                     dstQueueFamilyIndex = vk::QueueFamilyIgnored;
};

struct RenderingInfo {
	vk::RenderingFlags                           flags      = {};
	vk::Rect2D                                   renderArea = {}; // Should be specified
	u32                                          layerCount = 1;
	u32                                          viewMask   = 0;
	std::span<vk::RenderingAttachmentInfo const> colorAttachments;
	vk::RenderingAttachmentInfo const&           depthAttachment   = {.imageView = {}}; // == no depth
	vk::RenderingAttachmentInfo const&           stencilAttachment = {.imageView = {}}; // == no stencil
};

struct Viewport {
	float x;
	float y;
	float width;
	float height;
	float minDepth = 0.0f;
	float maxDepth = 1.0f;
};

class CommandBuffer : public vk::CommandBuffer {
public:
	using vk::CommandBuffer::CommandBuffer;
	using vk::CommandBuffer::operator=;
	CommandBuffer(vk::CommandBuffer const& other) : vk::CommandBuffer(other) {}

	void Barrier(vk::MemoryBarrier2 const& barrier = {});
	void Barrier(BufferBarrier const& barrier);
	void Barrier(ImageBarrier const& barrier);
	void Barrier(std::span<vk::MemoryBarrier2 const> const barriers);

	void BeginRendering(RenderingInfo const& info);
	void SetViewport(Viewport const& viewport);
	void SetScissor(vk::Rect2D const& scissor);
};

} // namespace VulkanRHI
