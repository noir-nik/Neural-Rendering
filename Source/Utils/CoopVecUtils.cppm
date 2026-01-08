export module NeuralGraphics:CoopVecUtils;
import std;
import vulkan_hpp;


export class CoopVecUtils {
public:
	using u32 = std::uint32_t;

	[[nodiscard]] static auto CalculateByteSize(
		vk::Device                                device,
		u32 const                                 rows,
		u32 const                                 cols,
		vk::CooperativeVectorMatrixLayoutNV const layout,
		vk::ComponentTypeKHR const                type) -> std::pair<vk::Result, std::size_t>;

	static constexpr auto GetMatrixAlignment() -> std::size_t { return kMatrixAlignment; }
	static constexpr auto GetVectorAlignment() -> std::size_t { return kBiasAlignment; }

private:
	static constexpr std::size_t kMatrixAlignment = 64;
	static constexpr std::size_t kBiasAlignment   = 16;
};
