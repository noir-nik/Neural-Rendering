module;
#include <cstddef>
#include <vulkan/vulkan.h>

module VulkanRHI;
import :CommandBuffer;
import vulkan_hpp;
import std;

using u32 = std::uint32_t;

namespace VulkanRHI {

void CommandBuffer::Barrier(vk::MemoryBarrier2 const& barrier) {
	vk::DependencyInfo const dependency{
		.memoryBarrierCount = 1,
		.pMemoryBarriers    = &barrier,
	};

	pipelineBarrier2(&dependency);
}

void CommandBuffer::Barrier(BufferBarrier const& barrier) {
	vk::BufferMemoryBarrier2 const barrier2{
		.pNext               = nullptr,
		.srcStageMask        = barrier.srcStageMask,
		.srcAccessMask       = barrier.srcAccessMask,
		.dstStageMask        = barrier.dstStageMask,
		.dstAccessMask       = barrier.dstAccessMask,
		.srcQueueFamilyIndex = barrier.srcQueueFamilyIndex,
		.dstQueueFamilyIndex = barrier.dstQueueFamilyIndex,
		.buffer              = barrier.buffer,
		.offset              = barrier.offset,
		.size                = barrier.size,
	};

	vk::DependencyInfo const dependency{
		.bufferMemoryBarrierCount = 1,
		.pBufferMemoryBarriers    = &barrier2,
	};

	pipelineBarrier2(&dependency);
}

void CommandBuffer::Barrier(ImageBarrier const& barrier) {
	vk::ImageSubresourceRange range{
		.aspectMask     = barrier.aspectMask,
		.baseMipLevel   = 0,
		.levelCount     = vk::RemainingMipLevels,
		.baseArrayLayer = 0,
		.layerCount     = vk::RemainingArrayLayers,
	};

	vk::ImageMemoryBarrier2 const barrier2{
		.pNext               = nullptr,
		.srcStageMask        = barrier.srcStageMask,
		.srcAccessMask       = barrier.srcAccessMask,
		.dstStageMask        = barrier.dstStageMask,
		.dstAccessMask       = barrier.dstAccessMask,
		.oldLayout           = barrier.oldLayout,
		.newLayout           = barrier.newLayout,
		.srcQueueFamilyIndex = barrier.srcQueueFamilyIndex,
		.dstQueueFamilyIndex = barrier.dstQueueFamilyIndex,
		.image               = barrier.image,
		.subresourceRange    = range,
	};

	vk::DependencyInfo const dependency{
		.imageMemoryBarrierCount = 1,
		.pImageMemoryBarriers    = &barrier2,
	};

	pipelineBarrier2(&dependency);
}

void CommandBuffer::Barrier(std::span<vk::MemoryBarrier2 const> const barriers) {
	vk::DependencyInfo const dependency{
		.memoryBarrierCount = static_cast<u32>(barriers.size()),
		.pMemoryBarriers    = barriers.data(),
	};

	pipelineBarrier2(&dependency);
}

void CommandBuffer::BeginRendering(RenderingInfo const& info) {
	vk::RenderingInfoKHR renderingInfo{
		.flags                = info.flags,
		.renderArea           = info.renderArea,
		.layerCount           = info.layerCount,
		.viewMask             = info.viewMask,
		.colorAttachmentCount = static_cast<u32>(info.colorAttachments.size()),
		.pColorAttachments    = info.colorAttachments.data(),
		.pDepthAttachment     = info.depthAttachment.imageView ? &info.depthAttachment : nullptr,
		.pStencilAttachment   = info.stencilAttachment.imageView ? &info.stencilAttachment : nullptr,
	};
	beginRendering(&renderingInfo);
}

void CommandBuffer::SetViewport(Viewport const& viewport) {
	vk::Viewport vkViewport {
		.x        = viewport.x,
		.y        = viewport.y,
		.width    = viewport.width,
		.height   = viewport.height,
		.minDepth = viewport.minDepth,
		.maxDepth = viewport.maxDepth,
	};
	setViewport(0, 1, &vkViewport);
}

void CommandBuffer::SetScissor(vk::Rect2D const& scissor) {
	vk::Rect2D vkScissor {
		.offset = { scissor.offset.x, scissor.offset.y },
		.extent = {
			.width  = scissor.extent.width,
			.height = scissor.extent.height
		}
	};
	setScissor(0, 1, &vkScissor);
}

} // namespace VulkanRHI
