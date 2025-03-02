module NeuralGraphics;
import :CoopVecUtils;
import std;
import vulkan_hpp;

namespace ng::inline Utils {
using u32 = std::uint32_t;

auto CoopVecUtils::CalculateByteSize(u32 const                                 rows,
									 u32 const                                 cols,
									 vk::CooperativeVectorMatrixLayoutNV const layout,
									 vk::ComponentTypeKHR const                type) -> std::pair<vk::Result, std::size_t> {
	std::size_t result_size = 0;
	std::size_t stride      = (layout == vk::CooperativeVectorMatrixLayoutNV::eRowMajor) ? cols * GetVulkanComponentSize(type) : rows * GetVulkanComponentSize(type);

	vk::ConvertCooperativeVectorMatrixInfoNV info{
		.srcSize          = 0,
		.srcData          = {.hostAddress = nullptr},
		.pDstSize         = &result_size,
		.dstData          = {.hostAddress = nullptr},
		.srcComponentType = type,
		.dstComponentType = type,
		.numRows          = rows,
		.numColumns       = cols,
		.srcLayout        = vk::CooperativeVectorMatrixLayoutNV::eRowMajor,
		.srcStride        = stride,
		.dstLayout        = layout,
		.dstStride        = stride,
	};

	vk::Result result = device.convertCooperativeVectorMatrixNV(&info);

	return {result, result_size};
}
} // namespace ng::inline Utils
