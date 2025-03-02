export module NeuralGraphics:CoopVecUtils;
import std;
import vulkan_hpp;

namespace ng::inline Utils {

class CoopVecUtils {
public:
	using u32 = std::uint32_t;
	CoopVecUtils(vk::Device device) : device(device) {}

	[[nodiscard]] auto CalculateByteSize(u32 const                                 rows,
										 u32 const                                 cols,
										 vk::CooperativeVectorMatrixLayoutNV const layout,
										 vk::ComponentTypeKHR const                type) -> std::pair<vk::Result, std::size_t>;

	auto GetMatrixAlignment() -> std::size_t { return kMatrixAlignment; }
	auto GetVectorAlignment() -> std::size_t { return kBiasAlignment; }

private:
	static const std::size_t kMatrixAlignment = 64;
	static const std::size_t kBiasAlignment   = 16;
	vk::Device               device;
};
} // namespace ng::inline Utils
