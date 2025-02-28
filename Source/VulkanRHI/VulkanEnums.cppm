export module VulkanRHI:Enums;
import vulkan_hpp;

export namespace VulkanRHI {
namespace Memory {
vk::MemoryPropertyFlags constexpr inline eGPU = vk::MemoryPropertyFlagBits::eDeviceLocal;
vk::MemoryPropertyFlags constexpr inline eCPU = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
} // namespace Memory
} // namespace VulkanRHI
