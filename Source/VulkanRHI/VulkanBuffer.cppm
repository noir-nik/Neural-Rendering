export module VulkanRHI:Buffer;
import :Enums;

import std;
import vulkan_hpp;
import vk_mem_alloc;

export namespace VulkanRHI {

struct BufferInfo {
	vk::DeviceSize          size;
	vk::BufferUsageFlags    usage;
	vk::MemoryPropertyFlags memory = Memory::eGPU;
	std::string_view        name   = "";
};

class Buffer : public vk::Buffer {
public:
	Buffer() = default;

	Buffer(Buffer const&)            = delete;
	Buffer& operator=(Buffer const&) = delete;
	Buffer(Buffer&& other) noexcept;
	Buffer& operator=(Buffer&& other) noexcept;

	// Destructor, frees resources
	~Buffer();

	// Create with result checked
	[[nodiscard]] auto Create(vk::Device device, VmaAllocator allocator, BufferInfo const& info) -> vk::Result;

	// Frees all resources
	void Destroy();

	// Get vk::Device address
	auto GetAddress() const -> vk::DeviceAddress;

	// Get mapped data pointer (Only CPU)
	// Doesn't require mapping or unmapping, can be called any number of times,
	auto GetMappedData() const -> void*;

	// Map buffer (Only CPU)
	// Buffer must be unmapped the same number of times as it was mapped
	auto Map() -> void*;

	// Unmap mapped buffer
	void Unmap();

	inline auto GetDevice() const -> vk::Device { return device; }
	inline auto GetAllocator() const -> VmaAllocator { return allocator; }
	inline auto GetAllocation() const -> VmaAllocation { return allocation; }
	inline auto GetAllocationInfo() const -> VmaAllocationInfo { return allocation_info; }
	inline auto GetSize() const -> vk::DeviceSize { return size; }
	inline auto GetUsage() const -> vk::BufferUsageFlags { return usage; }
	inline auto GetMemory() const -> vk::MemoryPropertyFlags { return memory; }

private:
	vk::Device device;

	VmaAllocator      allocator;
	VmaAllocation     allocation;
	VmaAllocationInfo allocation_info;

	vk::DeviceSize          size;
	vk::BufferUsageFlags    usage;
	vk::MemoryPropertyFlags memory;
};

} // namespace VulkanRHI
