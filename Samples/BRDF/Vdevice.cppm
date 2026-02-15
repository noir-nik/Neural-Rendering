module;
 
#include "Log.h"
#include <cstdio>

export module Vdevice;
import vulkan_hpp;

export struct VDevice : vk::Device {
	using Base = vk::Device;

	VDevice(Base base = {}) : Base(base) {}

	# define _FF(name) auto name(auto... args) { LOG_DEBUG("VDevice::" #name "()"); return Base::name(args...); }

	_FF(destroy);
	_FF(waitIdle);
	_FF(allocateMemory);
	_FF(freeMemory);
	_FF(mapMemory);
	_FF(unmapMemory);
	_FF(flushMappedMemoryRanges);
	_FF(invalidateMappedMemoryRanges);
	_FF(getMemoryCommitment);
	_FF(bindBufferMemory);
	_FF(bindImageMemory);
	_FF(getBufferMemoryRequirements);
	_FF(getImageMemoryRequirements);
	_FF(getImageSparseMemoryRequirements);
	_FF(createFence);
	_FF(destroyFence);
	_FF(resetFences);
	_FF(getFenceStatus);
	_FF(waitForFences);
	_FF(createSemaphore);
	_FF(destroySemaphore);
	_FF(createEvent);
	_FF(destroyEvent);
	_FF(getEventStatus);
	_FF(setEvent);
	_FF(resetEvent);
	_FF(createQueryPool);
	_FF(destroyQueryPool);
	_FF(getQueryPoolResults);
	_FF(createBuffer);
	_FF(destroyBuffer);
	_FF(createBufferView);
	_FF(destroyBufferView);
	_FF(createImage);
	_FF(destroyImage);
	_FF(getImageSubresourceLayout);
	_FF(createImageView);
	_FF(destroyImageView);
	_FF(createShaderModule);
	_FF(destroyShaderModule);
	_FF(createPipelineCache);
	_FF(destroyPipelineCache);
	_FF(getPipelineCacheData);
	_FF(mergePipelineCaches);
	_FF(createGraphicsPipelines);
	_FF(createComputePipelines);
	_FF(destroyPipeline);
	_FF(createPipelineLayout);
	_FF(destroyPipelineLayout);
	_FF(createSampler);
	_FF(destroySampler);
	_FF(createDescriptorSetLayout);
	_FF(destroyDescriptorSetLayout);
	_FF(createDescriptorPool);
	_FF(destroyDescriptorPool);
	_FF(resetDescriptorPool);
	_FF(allocateDescriptorSets);
	_FF(freeDescriptorSets);
	_FF(updateDescriptorSets);
	_FF(createFramebuffer);
	_FF(destroyFramebuffer);
	_FF(createRenderPass);
	_FF(destroyRenderPass);
	_FF(getRenderAreaGranularity);
	_FF(createCommandPool);
	_FF(destroyCommandPool);
	_FF(resetCommandPool);
	_FF(allocateCommandBuffers);
	_FF(freeCommandBuffers);
 
};