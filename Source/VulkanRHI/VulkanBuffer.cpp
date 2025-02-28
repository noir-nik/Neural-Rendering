module;

#include <vulkan/vulkan.h>

module VulkanRHI;
import :Buffer;
import :Enums;
import vulkan_hpp;
import vk_mem_alloc;
import std;

namespace VulkanRHI {

Buffer::Buffer(Buffer&& other) noexcept
	: vk::Buffer(std::exchange(static_cast<vk::Buffer&>(other), {})),
	  device(std::move(other.device)),
	  allocator(std::move(other.allocator)),
	  allocation(std::move(other.allocation)),
	  allocation_info(std::move(other.allocation_info)),
	  size(std::move(other.size)),
	  usage(std::move(other.usage)),
	  memory(std::move(other.memory)) {}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
	if (this != &other) {
		vk::Buffer::operator=(std::exchange(static_cast<vk::Buffer&>(other), {}));
		device          = std::move(other.device);
		allocator       = std::move(other.allocator);
		allocation      = std::move(other.allocation);
		allocation_info = std::move(other.allocation_info);
		size            = std::move(other.size);
		usage           = std::move(other.usage);
		memory          = std::move(other.memory);
	}
	return *this;
}

Buffer::~Buffer() { Destroy(); }

auto Buffer::GetMappedData() const -> void* {
	// ASSERT(memory & Memory::eCPU, "Buffer not cpu accessible!");
	return allocation_info.pMappedData;
}

auto Buffer::GetAddress() const -> vk::DeviceAddress {
	return GetDevice().getBufferAddress({.buffer = *this});
}

auto Buffer::Map() -> void* {
	// ASSERT(memory & Memory::eCPU, "Buffer not cpu accessible!");
	void* data;
	vmaMapMemory(GetAllocator(), allocation, &data);
	return data;
}

void Buffer::Unmap() {
	// ASSERT(memory & Memory::eCPU, "Buffer not cpu accessible!");
	vmaUnmapMemory(GetAllocator(), allocation);
}

auto Buffer::Create(vk::Device device, VmaAllocator allocator, BufferInfo const& info) -> vk::Result {
	this->device    = device;
	this->allocator = allocator;
	this->usage     = info.usage;
	this->memory    = info.memory;

	this->size = info.size; // + info.size % GetDevice().GetPhysicalDevice().GetProperties10().limits.minStorageBufferOffsetAlignment;

	vk::BufferCreateInfo bufferInfo{
		.size        = size,
		.usage       = usage,
		.sharingMode = vk::SharingMode::eExclusive,
	};

	using ::VmaAllocationCreateFlagBits;
	VmaAllocationCreateFlags constexpr kBufferCpuFlags =
		VMA_ALLOCATION_CREATE_MAPPED_BIT |
		VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

	VmaAllocationCreateInfo allocInfo = {
		.flags = memory & Memory::eCPU ? kBufferCpuFlags : VmaAllocationCreateFlags{},
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	vk::Result result =
		vk::Result(vmaCreateBuffer(GetAllocator(), &reinterpret_cast<VkBufferCreateInfo&>(bufferInfo), &allocInfo,
								   reinterpret_cast<VkBuffer*>(static_cast<vk::Buffer*>(this)), &allocation, &allocation_info));

	return result;
}

void Buffer::Destroy() {
	if (vk::Buffer::operator bool()) {
		vmaDestroyBuffer(GetAllocator(), *this, allocation);
		vk::Buffer::operator=(vk::Buffer{});
	}
}
} // namespace VulkanRHI
