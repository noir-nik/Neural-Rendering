module;

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

export module vk_mem_alloc;
export {
// NOLINTBEGIN(misc-unused-using-decls)
using ::VmaAllocationCreateFlagBits;
using ::VmaAllocatorCreateFlagBits;
using ::VmaDefragmentationFlagBits;
using ::VmaDefragmentationMoveOperation;
using ::VmaMemoryUsage;
using ::VmaPoolCreateFlagBits;
using ::VmaVirtualAllocationCreateFlagBits;
using ::VmaVirtualBlockCreateFlagBits;

using ::VmaAllocationCreateFlags;
using ::VmaAllocatorCreateFlags;
using ::VmaDefragmentationFlags;
using ::VmaPoolCreateFlags;
using ::VmaVirtualAllocationCreateFlags;
using ::VmaVirtualBlockCreateFlags;

using ::VmaAllocationCreateInfo;
using ::VmaAllocationInfo;
using ::VmaAllocationInfo2;
using ::VmaAllocatorCreateInfo;
using ::VmaAllocatorInfo;
using ::VmaBudget;
using ::VmaDefragmentationInfo;
using ::VmaDefragmentationMove;
using ::VmaDefragmentationPassMoveInfo;
using ::VmaDefragmentationStats;
using ::VmaDetailedStatistics;
using ::VmaDeviceMemoryCallbacks;
using ::VmaPoolCreateInfo;
using ::VmaStatistics;
using ::VmaTotalStatistics;
using ::VmaVirtualAllocationCreateInfo;
using ::VmaVirtualAllocationInfo;
using ::VmaVirtualBlockCreateInfo;
using ::VmaVulkanFunctions;

using ::VmaAllocation;
using ::VmaAllocator;
using ::VmaDefragmentationContext;
using ::VmaPool;
using ::VmaVirtualBlock;

using ::VmaAllocHandle;
using ::VmaVirtualAllocation;

using ::vmaAllocateMemory;
using ::vmaAllocateMemoryForBuffer;
using ::vmaAllocateMemoryForImage;
using ::vmaAllocateMemoryPages;
using ::vmaBeginDefragmentation;
using ::vmaBeginDefragmentationPass;
using ::vmaBindBufferMemory;
using ::vmaBindBufferMemory2;
using ::vmaBindImageMemory;
using ::vmaBindImageMemory2;
using ::vmaBuildStatsString;
using ::vmaBuildVirtualBlockStatsString;
using ::vmaCalculatePoolStatistics;
using ::vmaCalculateStatistics;
using ::vmaCalculateVirtualBlockStatistics;
using ::vmaCheckCorruption;
using ::vmaCheckPoolCorruption;
using ::vmaClearVirtualBlock;
using ::vmaCopyAllocationToMemory;
using ::vmaCopyMemoryToAllocation;
using ::vmaCreateAliasingBuffer;
using ::vmaCreateAliasingBuffer2;
using ::vmaCreateAliasingImage;
using ::vmaCreateAliasingImage2;
using ::vmaCreateAllocator;
using ::vmaCreateBuffer;
using ::vmaCreateBufferWithAlignment;
using ::vmaCreateImage;
using ::vmaCreatePool;
using ::vmaCreateVirtualBlock;
using ::vmaDestroyAllocator;
using ::vmaDestroyBuffer;
using ::vmaDestroyImage;
using ::vmaDestroyPool;
using ::vmaDestroyVirtualBlock;
using ::vmaEndDefragmentation;
using ::vmaEndDefragmentationPass;
using ::vmaFindMemoryTypeIndex;
using ::vmaFindMemoryTypeIndexForBufferInfo;
using ::vmaFindMemoryTypeIndexForImageInfo;
using ::vmaFlushAllocation;
using ::vmaFlushAllocations;
using ::vmaFreeMemory;
using ::vmaFreeMemoryPages;
using ::vmaFreeStatsString;
using ::vmaFreeVirtualBlockStatsString;
using ::vmaGetAllocationInfo;
using ::vmaGetAllocationInfo2;
using ::vmaGetAllocationMemoryProperties;
using ::vmaGetAllocatorInfo;
using ::vmaGetHeapBudgets;
using ::vmaGetMemoryProperties;
using ::vmaGetMemoryTypeProperties;
using ::vmaGetPhysicalDeviceProperties;
using ::vmaGetPoolName;
using ::vmaGetPoolStatistics;
using ::vmaGetVirtualAllocationInfo;
using ::vmaGetVirtualBlockStatistics;
using ::vmaInvalidateAllocation;
using ::vmaInvalidateAllocations;
using ::vmaIsVirtualBlockEmpty;
using ::vmaMapMemory;
using ::vmaSetAllocationName;
using ::vmaSetAllocationUserData;
using ::vmaSetCurrentFrameIndex;
using ::vmaSetPoolName;
using ::vmaSetVirtualAllocationUserData;
using ::vmaUnmapMemory;
using ::vmaVirtualAllocate;
using ::vmaVirtualFree;
// NOLINTEND(misc-unused-using-decls)
}
