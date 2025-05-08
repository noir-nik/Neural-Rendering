module NeuralGraphics;
import :CoopVecUtils;
import std;
import vulkan_hpp;

using u32 = std::uint32_t;
using namespace Utils;

auto CoopVecUtils::CalculateByteSize(
		vk::Device                                device,
		u32 const                                 rows,
		u32 const                                 cols,
		vk::CooperativeVectorMatrixLayoutNV const layout,
		vk::ComponentTypeKHR const                type) -> std::pair<vk::Result, std::size_t> {
	std::size_t result_size = 0;
	std::size_t src_stride      =  cols * GetVulkanComponentSize(type);
	std::size_t dst_stride      = (layout == vk::CooperativeVectorMatrixLayoutNV::eRowMajor) ? cols * GetVulkanComponentSize(type) : rows * GetVulkanComponentSize(type);

	// std::printf("stride: %zu, rows: %u, cols: %u, component size: %zu\n", src_stride, rows, cols, GetVulkanComponentSize(type));

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
		.srcStride        = src_stride,
		.dstLayout        = layout,
		.dstStride        = dst_stride,
	};

	vk::Result result = device.convertCooperativeVectorMatrixNV(&info);

	return {result, result_size};
}
