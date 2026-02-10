#include "Shaders/BRDFBindings.h"
module;
#include "CheckResult.h"
#include <cassert> // assert

module BRDFSample;
import NeuralGraphics;
import vulkan_hpp;
import WeightsLoader;
import SamplesCommon;
import Math;
import std;
import FastKan;

import nlohmann.json;
using json = nlohmann::json;

using numeric::float16_t;

void DumpVertexData(std::span<const UVSphere::Vertex> vertices, std::span<const UVSphere::IndexType> indices) {
	std::printf("Vertices:\n");
	for (u32 i = 0; i < vertices.size(); ++i) {
		const auto& vertex = vertices[i];
		std::printf("%u) pos = (%f, %f, %f), uv = (%f, %f), normal = (%f, %f, %f)\n", i, vertex.pos[0], vertex.pos[1], vertex.pos[2], vertex.u, vertex.v, vertex.normal[0], vertex.normal[1], vertex.normal[2]);
	}
	std::printf("Indices:\n");
	for (u32 i = 0; i < indices.size() / 3; ++i) {
		std::printf("%u) %u, %u, %u\n", i, indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]);
	}
}

template <typename T>
	requires(std::is_same_v<T, float> || std::is_same_v<T, float16_t>)
auto PrintLayerBiases(Linear layer, std::byte const* parameters) -> void {
	if (!parameters) return;
	auto const* p_bias = reinterpret_cast<T const*>(parameters + layer.GetBiasesOffset());

	for (u32 i = 0; i < layer.GetBiasesCount(); ++i) {
		std::printf("%10.6f ", float(p_bias[i]));
	}
	std::printf("\n");
}

// void BRDFSample::ReadKANWeights(NetworkBufferInfo const& network_info) {}

// std::pair<nlohmann::json, std::vector<std::map<std::string, std::vector<float>>>>

template <typename... Args>
auto make_string(Args&&... args)
	requires(std::is_convertible_v<Args, std::string_view> && ...)
{
	using SV = std::string_view;

	// std::tuple const t(SV(std::forward<Args>(args))...);

	std::size_t total_size = 0;
	((total_size += SV(std::forward<Args>(args)).size()), ...);
	std::string result;
	result.resize(total_size);

	std::size_t offset = 0;
	((std::copy(SV(std::forward<Args>(args)).cbegin(), SV(std::forward<Args>(args)).cend(), result.begin() + offset), offset += SV(std::forward<Args>(args)).size()), ...);
	return result;
}

using ReadKANResult = std::expected<FastKan, std::string>;
[[nodiscard]] auto ReadKANWeights(std::string_view file_name) -> ReadKANResult {
	using namespace nlohmann;

	std::string prefix = std::string(file_name);
	// std::string prefix = (network_info.file_name);
	if (prefix.ends_with(".json")) {
		prefix = prefix.substr(0, prefix.size() - 5);
	} else if (prefix.ends_with(".bin")) {
		prefix = prefix.substr(0, prefix.size() - 4);
	}

	auto json_name = prefix + ".json";

	std::ifstream json_file(json_name);
	if (!json_file.is_open()) {
		// std::cerr << "Error: Could not open " << prefix << ".json" << std::endl;
		// std::exit(1);
		return std::unexpected(make_string("Could not open ", json_name));
	}

	json json_manifest;
	json_file >> json_manifest;

	std::string dtype_str = json_manifest["dtype"];
	enum class DataType {
		Float32,
		Float16,
	};
	DataType dtype;
	if (dtype_str == "float32") {
		dtype = DataType::Float32;
	} else {
		// std::cerr << "Error: Unsupported data type " << dtype_str << std::endl;
		// std::exit(1);
		// return std::unexpected(std::string("Unsupported data type ") + dtype_str);
		return std::unexpected(make_string("Unsupported data type ", dtype_str));
	}

	std::ifstream bin_file(prefix + ".bin", std::ios::binary);
	if (!bin_file.is_open()) {
		// std::cerr << "Error: Could not open " << prefix << ".bin" << std::endl;
		// std::exit(1);
		return std::unexpected(make_string("Could not open ", prefix, ".bin"));
	}

	std::vector<float> bin_data; //((std::istreambuf_iterator<std::streambuf>(bin_file)), std::istreambuf_iterator<std::streambuf>());
	bin_file.seekg(0, std::ios::end);
	bin_data.resize(std::streamsize(bin_file.tellg()));
	bin_file.seekg(0);
	bin_file.read(reinterpret_cast<char*>(bin_data.data()), bin_data.size());
	bin_file.close();

	FastKan kan;

	auto const& json_layers = json_manifest["layers"];
	kan.layers_.reserve(json_layers.size());

	for (const auto& json_layer : json_layers) {
		FastKanLayer kan_layer;
		for (const auto& param : json_layer["params"].items()) {
			auto const& name   = param.key(); // param.first;
			auto const& info   = param.value();
			u32         offset = info["offset"];
			u32         size   = info["size"];
			auto        shape  = info["shape"];

			// auto start = bin_data.data() + offset;

			auto ldata = KANBuffer(offset, size);

			// shape
			// for (u32 i = 0; i < shape.size(); ++i) {
			// 	ldata.shape_[ldata.shape_size_++] = shape[i];
			// }

			ldata.set_shape(shape);

			if (name == "rbf_grid") {
				kan_layer.rbf_grid() = ldata;
			} else if (name == "rbf_denom_inv") {
				kan_layer.rbf_denom_inv() = ldata;
			} else if (name == "spline_weight") {
				kan_layer.spline_weight() = ldata;
			} else if (name == "base_weight") {
				kan_layer.base_weight() = ldata;
			} else if (name == "base_bias") {
				kan_layer.base_bias() = ldata;
			}
		}
		kan.layers_.push_back(kan_layer);
	}
	kan.buffer_ = std::move(bin_data);

	// return {std::move(json_manifest), std::move(layers)};
	return kan;
}

using ComponentTy = vk::ComponentTypeKHR;
using LayoutTy    = vk::CooperativeVectorMatrixLayoutNV;

struct MatrixInfo {
	vk::Device  device;
	void const* src;
	std::size_t src_size;
	std::byte*  dst;
	LayoutTy    dst_layout;
	ComponentTy src_component_type;
	ComponentTy dst_matrix_type;
	u32         rows;
	u32         cols;
};

auto write_matrix_kan(MatrixInfo const& info) -> std::size_t {
	std::size_t required_size;

	auto const& [device, src, src_size, dst, dst_layout, src_component_type, dst_matrix_type, rows, cols] = info;

	vk::ConvertCooperativeVectorMatrixInfoNV create_info{
		.srcSize          = info.src_size,
		.srcData          = {.hostAddress = info.src},
		.pDstSize         = &required_size,
		.dstData          = {.hostAddress = info.dst},
		.srcComponentType = info.src_component_type,
		.dstComponentType = info.dst_matrix_type,
		.numRows          = info.rows,
		.numColumns       = info.cols,
		.srcLayout        = vk::CooperativeVectorMatrixLayoutNV::eRowMajor,
		.srcStride        = info.cols * Utils::GetVulkanComponentSize(info.src_component_type),
		.dstLayout        = info.dst_layout,
		.dstStride        = info.cols * Utils::GetVulkanComponentSize(info.dst_matrix_type),
	};

	CHECK_VULKAN_RESULT(info.device.convertCooperativeVectorMatrixNV(&create_info));

	std::printf("Src size: %zu, result size: %zu\n", info.src_size, required_size);

	return required_size;
};

void write_matrix_mlp(
	vk::Device    device,
	void const*   src,
	std::size_t   src_size,
	std::byte*    dst,
	LayoutTy      dst_layout,
	ComponentTy   src_component_type,
	ComponentTy   dst_matrix_type,
	Linear const& linear) {
	std::size_t expected_size = linear.GetWeightsSize();
	std::size_t required_size = expected_size;

	vk::ConvertCooperativeVectorMatrixInfoNV info{
		.srcSize          = src_size,
		.srcData          = {.hostAddress = src},
		.pDstSize         = &required_size,
		.dstData          = {.hostAddress = dst + linear.GetWeightsOffset()},
		.srcComponentType = src_component_type,
		.dstComponentType = dst_matrix_type,
		.numRows          = linear.GetOutputsCount(),
		.numColumns       = linear.GetInputsCount(),
		.srcLayout        = vk::CooperativeVectorMatrixLayoutNV::eRowMajor,
		.srcStride        = linear.GetInputsCount() * Utils::GetVulkanComponentSize(src_component_type),
		.dstLayout        = dst_layout,
		.dstStride        = linear.GetInputsCount() * Utils::GetVulkanComponentSize(dst_matrix_type),
	};

	info.dstData.hostAddress = nullptr;
	CHECK_VULKAN_RESULT(device.convertCooperativeVectorMatrixNV(&info));
	if (required_size != expected_size) {
		std::printf("Expected size: %zu, actual size: %zu\n", expected_size, required_size);
		std::exit(1);
	}
	info.dstData.hostAddress = dst + linear.GetWeightsOffset();
	CHECK_VULKAN_RESULT(device.convertCooperativeVectorMatrixNV(&info));
};

// struct

template <typename SrcBiasType, typename DstBiasType>
void write_network(
	vk::Device                  device,
	VulkanCoopVecNetwork const& network,
	std::byte*                  src_parameters,
	std::byte*                  dst_parameters,
	LayoutTy                    dst_layout,
	ComponentTy                 src_component_type,
	ComponentTy                 dst_matrix_type) {
	auto src_offset = std::size_t{0};
	for (u32 i = 0; i < network.GetLayers().size(); ++i) {
		auto& layer = network.GetLayer<Linear>(i);

		std::size_t const src_weights_size_bytes = layer.GetWeightsCount() * Utils::GetVulkanComponentSize(src_component_type);
		std::size_t const src_biases_size_bytes  = layer.GetBiasesCount() * Utils::GetVulkanComponentSize(src_component_type);

		auto const* src_weights = src_parameters + src_offset;
		write_matrix_mlp(device, src_weights, src_weights_size_bytes, dst_parameters, dst_layout, src_component_type, dst_matrix_type, layer);
		src_offset += src_weights_size_bytes;

		auto const* src_bias = src_parameters + src_offset;
		std::byte*  dst_bias = dst_parameters + layer.GetBiasesOffset();
		// std::memcpy(dst_bias, src_bias, src_biases_size_bytes);
		for (u32 j = 0; j < layer.GetBiasesCount(); ++j) {
			DstBiasType* p_dst = reinterpret_cast<DstBiasType*>(dst_bias + j * sizeof(DstBiasType));
			*p_dst             = static_cast<DstBiasType>(reinterpret_cast<SrcBiasType const*>(src_bias)[j]);
		}
		src_offset += src_biases_size_bytes;
	}
};

// std::vector<FastKanLayer> layers_;
// auto constexpr kVectorAlignment = std::size_t{sizeof(float) * 4};
auto constexpr kVectorAlignment = CoopVecUtils::GetVectorAlignment();

struct FastKanLayerOffsets {
	FastKanLayerBase<u64> offsets;
	std::size_t           total_size = 0;
};

// using FastKanLayerOffsets = FastKanLayerBase<u64>;

auto write_fast_kan_layer(
	vk::Device             device,
	FastKanLayer const&    layer,
	std::span<float const> src_buffer,
	std::byte*             dst_parameters,
	LayoutTy               dst_layout,
	ComponentTy            src_component_type,
	ComponentTy            dst_matrix_type)
	-> FastKanLayerOffsets {

	FastKanLayerOffsets result;

	auto& offset = result.total_size;
	// auto offset = std::size_t{0};

	auto& buffer_offsets = result.offsets;

	auto wrt = [&](KANBuffer const& buffer) {
		auto sspan    = buffer.span(src_buffer.data());
		auto dstns    = std::distance(&layer.get_buffers()[0], &buffer);
		auto dst_size = std::size_t{buffer.size() * sizeof(float16_t)};
		std::printf("Writing buffer %s %td/%zu, offset: %zu, size: %zu, dst_size: %zu\n", layer.get_buffer_name(dstns).data(), dstns, layer.get_buffers().size(), offset, sspan.size_bytes(), dst_size);
		// write_fast_kan_buffer(buffer, dst_parameters + offset);
		{
			if (buffer.offset() + buffer.size() > src_buffer.size()) {
				std::printf("Buffer size too small 1\n");
				std::printf("Buffer offset: %zu, buffer size: %zu, src_buffer size: %zu\n", buffer.offset(), buffer.size(), src_buffer.size());
				std::exit(1);
			}
			std::byte* dst = dst_parameters + offset;
			// std::memcpy(dst, sspan.data(), sspan.size_bytes());
			// float -> float16
			for (u32 i = 0; i < buffer.size(); ++i) {
				*(reinterpret_cast<float16_t*>(dst) + i) = static_cast<float16_t>(sspan[i]);
			}
		}
		return AlignUpPowerOfTwo(offset + dst_size, kVectorAlignment);
	};

	// std::printf("Writing rbf_grid, offset: %zu\n", offset);
	buffer_offsets.rbf_grid() = offset;
	offset                    = wrt(layer.rbf_grid());
	// std::printf("Writing rbf_denom_inv, offset: %zu\n", offset);
	buffer_offsets.rbf_denom_inv() = offset;
	offset                         = wrt(layer.rbf_denom_inv());
	// std::printf("Writing base_bias, offset: %zu\n", offset);
	buffer_offsets.base_bias() = offset;
	offset                     = wrt(layer.base_bias());
	offset                     = AlignUpPowerOfTwo(offset, CoopVecUtils::GetMatrixAlignment());

	// ret.get_spline_weight() = wrt(layer.get_spline_weight());
	// ret.get_base_weight()   = wrt(layer.get_base_weight());
	// for (u32 i = 0; auto const& buffer : {layer.get_rbf_grid(), layer.get_rbf_denom_inv(), layer.get_base_bias()})

	// Write 2 matrices: [spline_weight base_weight]

	// auto const base_inputs = layer.get_base_weight().shape()[1];
	// auto const base_ouputs = layer.get_base_weight().shape()[0];
	// auto const base_cols   = base_inputs;
	// auto const base_rows   = base_ouputs;

	std::size_t matrix_size;
	std::printf("Writing spline_weight, offset: %zu\n", offset);
	matrix_size = write_matrix_kan({
		.device             = device,
		.src                = layer.spline_weight().span(src_buffer.data()).data(),
		.src_size           = layer.spline_weight().size_bytes(),
		.dst                = dst_parameters + offset,
		.dst_layout         = dst_layout,
		.src_component_type = src_component_type,
		.dst_matrix_type    = dst_matrix_type,
		.rows               = layer.spline_weight().shape()[0],
		.cols               = layer.spline_weight().shape()[1],
	});

	buffer_offsets.spline_weight() = offset;
	offset                         = AlignUpPowerOfTwo(offset + matrix_size, CoopVecUtils::GetMatrixAlignment());

	std::printf("Writing base_weight, offset: %zu\n", offset);
	matrix_size = write_matrix_kan({
		.device             = device,
		.src                = layer.base_weight().span(src_buffer.data()).data(),
		.src_size           = layer.base_weight().size_bytes(),
		.dst                = dst_parameters + offset,
		.dst_layout         = dst_layout,
		.src_component_type = src_component_type,
		.dst_matrix_type    = dst_matrix_type,
		.rows               = layer.base_weight().shape()[0],
		.cols               = layer.base_weight().shape()[1],
	});

	buffer_offsets.base_weight() = offset;
	offset                       = AlignUpPowerOfTwo(offset + matrix_size, CoopVecUtils::GetMatrixAlignment());

	return result;
}

auto write_fast_kan(
	vk::Device     device,
	FastKan const& kan,
	std::byte*     dst_parameters,
	LayoutTy       dst_layout,
	ComponentTy    src_component_type,
	ComponentTy    dst_matrix_type)
	-> FastKanOffsets {
	auto layers = kan.layers();

	auto base_offset = std::size_t{0};

	FastKanOffsets result;
	result.reserve(layers.size());

	for (auto& layer : layers) {
		std::printf("Writing layer, offset: %zu\n", base_offset);
		auto buffer_offsets = write_fast_kan_layer(device, layer, kan.buffer(), dst_parameters + base_offset, dst_layout, src_component_type, dst_matrix_type);
		for (u32 i = 0; i < 5; ++i) {
			buffer_offsets.offsets.get_buffer(i) += base_offset;
		}

		result.push_back(buffer_offsets.offsets);
		base_offset += buffer_offsets.total_size;
	}

	return result;
}

struct CubeMetadata {
	static constexpr std::size_t kNumFaces = 6;

	std::array<unsigned char*, kNumFaces> cubemap;

	int  width{0};
	int  height{0};
	int  channels{0};
	auto image_size() const -> std::size_t { return width * height * channels; }
	auto image_size_bytes() const -> std::size_t { return image_size() * sizeof(*cubemap[0]); }

	auto size() const -> std::size_t { return image_size() * kNumFaces; }
	auto size_bytes() const -> std::size_t { return image_size_bytes() * kNumFaces; }
	auto images() const -> std::span<const unsigned char* const> { return cubemap; }
};

auto LoadCubemap(std::string_view env_map_folder_path) -> CubeMetadata {
	// Load all the textures.

	int width{0};
	int height{0};
	int channels{0};

	char buf[512];

	auto concat_str = [&](std::string_view a, std::string_view b) -> std::string_view {
		auto const result = std::snprintf(buf, sizeof(buf), "%s/%s", a.data(), b.data());
		return {buf, static_cast<size_t>(result)};
	};

	auto loads = [&](std::string_view filename) -> unsigned char* {
		auto const path     = concat_str(env_map_folder_path, filename);
		std::array old_data = {width, height, channels};
		auto       res      = stbi_load(path.data(), &width, &height, &channels, 4);
		// verify no difference old_data - new_data
		auto [old_width, old_height, old_channels] = old_data;
		{
			std::array new_data   = {width, height, channels};
			std::array data_names = {"width", "height", "channels"};
			auto       get_data   = [&](u32 i) { return std::tuple{old_data[i], new_data[i], data_names[i]}; };
			if (std::all_of(old_data.begin(), old_data.end(), [](auto const& i) { return i != 0; })) {
				for (u32 i = 0; i < old_data.size(); ++i) {
					auto [old, new_, name] = get_data(i);
					if (old_data[i] != new_data[i]) {
						std::printf("Cubemap size mismatch %s Old: %u, New: %u\n", name, old, new_);
						std::exit(1);
					}
				}
			}
		}
		return res;
	};

	std::array cube_data = {
		loads("nx.png"),
		loads("ny.png"),
		loads("nz.png"),
		loads("px.png"),
		loads("py.png"),
		loads("pz.png"),
	};
	return {cube_data, width, height, channels};
}

void BRDFSample::CreateAndUploadBuffers(NetworkBufferInfo const& network_info) {
	sphere = UVSphere(1.0f, 32 * 2, 16 * 2);

	std::size_t vertices_size_bytes = sphere.GetVertexCount() * sizeof(UVSphere::Vertex);
	std::size_t indices_size_bytes  = sphere.GetIndexCount() * sizeof(UVSphere::IndexType);
	// std::size_t alignment             = kVectorAlignment;
	std::size_t vertices_size_aligned = AlignUpPowerOfTwo(vertices_size_bytes, kVectorAlignment);
	std::size_t indices_size_aligned  = AlignUpPowerOfTwo(indices_size_bytes, kVectorAlignment);

	// ENABLE_FAST_KAN
	FastKan const kan; /* = *ReadKANWeights(kan_weights_file_name).or_else([](auto&& error) -> ReadKANResult {
		std::cerr << error << std::endl;
		std::exit(1);
		return {};
	});

	kan.repr();
	// for (auto const& l : kan.layers())
	// 	l.repr_buffer(kan.buffer());

	std::printf("Total size: %zu\n", kan.size());
 */

	std::vector<float>        brdf_weights_vec;
	std::vector<LayerVariant> layers;
	CHECK(load_weights(network_info.file_name.data(), layers, brdf_weights_vec, network_info.header.data()));

	// if (layers.size() != expected_layer_count) {
	// 	std::printf("Error loading weights : wrong number of layers\n");
	// 	std::exit(1);
	// }

	// Get total size for buffer
	networks[u32(BrdfFunctionType::eWeightsInBuffer)].Init(layers);
	networks[u32(BrdfFunctionType::eWeightsInBufferF16)].Init(layers);
	networks[u32(BrdfFunctionType::eCoopVec)].Init(layers);
	CHECK_VULKAN_RESULT(networks[u32(BrdfFunctionType::eWeightsInBuffer)].UpdateOffsetsAndSize(device, LayoutTy::eRowMajor, ComponentTy::eFloat32, ComponentTy::eFloat32));
	CHECK_VULKAN_RESULT(networks[u32(BrdfFunctionType::eWeightsInBufferF16)].UpdateOffsetsAndSize(device, LayoutTy::eRowMajor, ComponentTy::eFloat16, ComponentTy::eFloat16));
	CHECK_VULKAN_RESULT(networks[u32(BrdfFunctionType::eCoopVec)].UpdateOffsetsAndSize(device, LayoutTy::eInferencingOptimal, ComponentTy::eFloat16, ComponentTy::eFloat16));

	CubeMetadata cube_data;

	if (hasattr(&BRDFSample::cubemap_folder_path)) {

		cube_data = LoadCubemap(cubemap_folder_path);

		cubemap_image.Create(
			device, vma_allocator, allocator,
			{.image_info = {
				 .flags         = vk::ImageCreateFlagBits::eCubeCompatible,
				 .imageType     = vk::ImageType::e2D,
				 .format        = vk::Format::eR8G8B8A8Unorm,
				 .extent        = {static_cast<u32>(cube_data.width), static_cast<u32>(cube_data.height), 1},
				 .mipLevels     = 1,
				 .arrayLayers   = kCubeSideCount,
				 .samples       = vk::SampleCountFlagBits::e1,
				 .tiling        = vk::ImageTiling::eOptimal,
				 .usage         = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
				 .sharingMode   = vk::SharingMode::eExclusive,
				 .initialLayout = vk::ImageLayout::eUndefined,
			 },
			 .aspect = vk::ImageAspectFlagBits::eColor});
	}

	vk::DeviceSize const kMinBufferSize = 256 * 1024;

	std::size_t total_size_bytes =
		std::max(
			kMinBufferSize,
			vertices_size_aligned + indices_size_aligned
				+ networks[u32(BrdfFunctionType::eCoopVec)].GetParametersSize()
				+ networks[u32(BrdfFunctionType::eWeightsInBuffer)].GetParametersSize()
				+ networks[u32(BrdfFunctionType::eWeightsInBufferF16)].GetParametersSize()
				+ kan.size_bytes() * 5
				+ cube_data.size_bytes() * 2);

	// Create buffers
	// clang-format off
	CHECK_VULKAN_RESULT(device_buffer.Create(device, vma_allocator, {
		.size   = total_size_bytes,
		.usage  = vk::BufferUsageFlagBits::eTransferDst 
				| vk::BufferUsageFlagBits::eStorageBuffer
				| vk::BufferUsageFlagBits::eVertexBuffer
				| vk::BufferUsageFlagBits::eIndexBuffer,
		.memory = vk::MemoryPropertyFlagBits::eDeviceLocal,
	}));
 
	CHECK_VULKAN_RESULT(staging_buffer.Create(device, vma_allocator, {
		.size   = total_size_bytes,
		.usage  = vk::BufferUsageFlagBits::eTransferSrc,
		.memory = vk::MemoryPropertyFlagBits::eHostVisible 
				| vk::MemoryPropertyFlagBits::eHostCoherent,
	}));
	// clang-format on

	vk::SamplerCreateInfo sampler_info{
		.flags            = {},
		.magFilter        = vk::Filter::eLinear,
		.minFilter        = vk::Filter::eLinear,
		.addressModeU     = vk::SamplerAddressMode::eClampToEdge,
		.addressModeV     = vk::SamplerAddressMode::eClampToEdge,
		.addressModeW     = vk::SamplerAddressMode::eClampToEdge,
		.mipLodBias       = 0.0f,
		// .anisotropyEnable = vk::True,
		.maxAnisotropy    = 16.0f,
		.compareEnable    = vk::False,
		.compareOp        = vk::CompareOp::eNever,
		.minLod           = 0.0f,
		.maxLod           = 1.0f,
		.borderColor      = vk::BorderColor::eFloatOpaqueWhite,
	};

	CHECK_VULKAN_RESULT(device.createSampler(&sampler_info, GetAllocator(), &cubemap_sampler));

	// Update descriptor set
	{
		vk::DescriptorBufferInfo buffer_infos[] = {{.buffer = device_buffer, .offset = 0, .range = device_buffer.GetSize()}};
		vk::DescriptorImageInfo  image_infos[]  = {{
			  .sampler     = cubemap_sampler,
			  .imageView   = cubemap_image.GetView(),
			  .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        }};

		vk::WriteDescriptorSet writes[] = {
			{
				.dstSet          = descriptor_set,
				.dstBinding      = BINDING_STORAGE_BUFFER,
				.dstArrayElement = 0,
				.descriptorCount = static_cast<u32>(std::size(buffer_infos)),
				.descriptorType  = vk::DescriptorType::eStorageBuffer,
				.pBufferInfo     = buffer_infos,
			},
			{
				.dstSet          = descriptor_set,
				.dstBinding      = BINDING_TEXTURE,
				.dstArrayElement = 0,
				.descriptorCount = static_cast<u32>(std::size(image_infos)),
				.descriptorType  = vk::DescriptorType::eCombinedImageSampler,
				.pImageInfo      = image_infos,
			},
		};

		auto descriptor_copy_count = 0u;
		auto copy_descriptor_sets  = nullptr;
		device.updateDescriptorSets(std::size(writes), writes, descriptor_copy_count, copy_descriptor_sets);
	}

	std::size_t offset    = 0;
	this->vertices_offset = offset;
	offset += vertices_size_aligned;
	this->indices_offset = offset;
	offset += indices_size_bytes;

	for (auto i = 1u; i < std::size(networks); ++i) {
		offset             = AlignUpPowerOfTwo(offset, CoopVecUtils::GetMatrixAlignment());
		weights_offsets[i] = offset;
		offset += AlignUpPowerOfTwo(networks[i].GetParametersSize(), CoopVecUtils::GetMatrixAlignment());
	}

	auto p_staging  = static_cast<std::byte*>(staging_buffer.GetMappedData());
	auto p_vertices = reinterpret_cast<UVSphere::Vertex*>(p_staging + this->vertices_offset);
	auto p_indices  = reinterpret_cast<UVSphere::IndexType*>(p_staging + this->indices_offset);
	sphere.WriteVertices(p_vertices);
	sphere.WriteIndices(p_indices);

	auto brdf_weights_src = std::span{reinterpret_cast<std::byte*>(brdf_weights_vec.data()), brdf_weights_vec.size() * sizeof(float)};

	write_network<float, numeric::float16_t>(device, networks[u32(BrdfFunctionType::eCoopVec)], brdf_weights_src.data(), p_staging + weights_offsets[u32(BrdfFunctionType::eCoopVec)], LayoutTy::eInferencingOptimal, ComponentTy::eFloat32, ComponentTy::eFloat16);
	write_network<float, float>(device, networks[u32(BrdfFunctionType::eWeightsInBuffer)], brdf_weights_src.data(), p_staging + weights_offsets[u32(BrdfFunctionType::eWeightsInBuffer)], LayoutTy::eRowMajor, ComponentTy::eFloat32, ComponentTy::eFloat32);
	write_network<float, numeric::float16_t>(device, networks[u32(BrdfFunctionType::eWeightsInBufferF16)], brdf_weights_src.data(), p_staging + weights_offsets[u32(BrdfFunctionType::eWeightsInBufferF16)], LayoutTy::eRowMajor, ComponentTy::eFloat32, ComponentTy::eFloat16);

	// std::printf("offset: %zu\n", offset);
	// std::printf("weights_offsets: %zu\n", weights_offsets[u32(BrdfFunctionType::eWeightsInBufferF16)] + networks[u32(BrdfFunctionType::eWeightsInBufferF16)].GetParametersSize());

	auto const fastkan_offset = offset;

	auto dst_layout = LayoutTy::eInferencingOptimal;
	// auto dst_layout = LayoutTy::eRowMajor;

	if (!kan_weights_file_name.empty()) {

		kan_offsets = write_fast_kan(device, kan, p_staging + offset, dst_layout, ComponentTy::eFloat32, ComponentTy::eFloat16);

		for (auto i = 0u; i < std::size(kan_offsets); ++i) {
			auto& offs = kan_offsets[i];
			offs.rbf_grid() += offset;
			offs.rbf_denom_inv() += offset;
			offs.spline_weight() += offset;
			offs.base_weight() += offset;
			offs.base_bias() += offset;
			if (0) {
				std::printf("kan_offset[%u]:\n", i);
				std::printf("rbf_grid: %zu\n", offs.rbf_grid());
				std::printf("rbf_denom_inv: %zu\n", offs.rbf_denom_inv());
				std::printf("spline_weight: %zu\n", offs.spline_weight());
				std::printf("base_weight: %zu\n", offs.base_weight());
				std::printf("base_bias: %zu\n", offs.base_bias());
			}
		}

		auto get_staging_kan = [&](std::size_t offset) -> float { return static_cast<float const>(*reinterpret_cast<float16_t const*>(p_staging + offset)); };

		auto verify_fastkan = [&](int layer_id, int buffer_id, int first_elem, int count) -> void {
			std::printf("verify_kan: %-20s\n", (kan.layers()[layer_id]).get_buffer_name(buffer_id).data());
			auto max_elems = kan.layers()[layer_id].get_buffer(buffer_id).size();
			for (int i = first_elem; i < std::min(decltype(max_elems)(count), max_elems); ++i) {
				auto const staging_val = get_staging_kan((kan_offsets[layer_id]).get_buffer(buffer_id) + i * sizeof(float16_t));
				auto const src_val     = (kan.layers()[layer_id]).get_buffer(buffer_id).span(kan.buffer().data())[first_elem + i];
				auto const always_verify
					// = true
					= false;
				auto const eps = 0.01f;
				if ((std::abs(src_val - staging_val) > eps) || always_verify) {
					std::printf("src_val: %f, staging_val: %f\n", src_val, staging_val);
					// std::printf("Error in layer %d, buffer %d, elem %d\n", layer_id, buffer_id, i);
					// std::exit(1);
				}
			}
		};
		for (int i = 0; i < kan.layers().size(); ++i) {
			for (int j = 0; j < kan.layers()[i].get_buffers().size(); ++j) {
				verify_fastkan(i, j, 0, 5);
			}
		}
	}
	if (verbose) {
		// networks[u32(BrdfFunctionType::eScalarBuffer)].Print();
		// networks[u32(BrdfFunctionType::eCoopVec)].Print();
		// PrintLayerBiases<float16_t>(networks[u32(BrdfFunctionType::eCoopVec)].GetLayer<Linear>(networks[u32(BrdfFunctionType::eCoopVec)].GetLayers().size() - 1), p_staging + this->optimal_weights_offset);
		// networks[u32(BrdfFunctionType::eScalarBuffer)].PrintLayerBiases(networks[u32(BrdfFunctionType::eScalarBuffer)].GetLayers().size() - 1, Component::eFloat32, p_staging + this->linear_weights_offset);
	}

	auto const cube_offset = offset;

	constexpr auto num_cube_sides = std::size(cube_data.cubemap);

	// u64 cube_offsets[num_cube_sides]{};

	// vk::BufferImageCopy cube_region{};
	vk::BufferImageCopy cube_regions[num_cube_sides];

	if (hasattr(&BRDFSample::cubemap_folder_path)) {
		auto const range = [](u32 num) { return std::views::iota(0u, num); };

		for (u32 const side_id : range(num_cube_sides)) {
			auto imdata = cube_data.cubemap[side_id];

			std::memcpy(p_staging + offset, imdata, cube_data.image_size_bytes());
			// cube_offsets[i] = offset;
			// auto const cube_region = vk::BufferImageCopy{
			cube_regions[side_id] = vk::BufferImageCopy{
				.bufferOffset      = offset,
				.bufferRowLength   = 0,
				.bufferImageHeight = 0,
				.imageSubresource  = vk::ImageSubresourceLayers{
					 .aspectMask     = vk::ImageAspectFlagBits::eColor,
					 .mipLevel       = 0,
					 .baseArrayLayer = side_id,
					 .layerCount     = 1,
                },
				.imageOffset = vk::Offset3D{0, 0, 0},
				.imageExtent = vk::Extent3D{static_cast<uint32_t>(cube_data.width), static_cast<uint32_t>(cube_data.height), 1},
			};
			offset = AlignUpPowerOfTwo(offset + cube_data.image_size_bytes(), kVectorAlignment);
		}
	}

	// DumpVertexData({p_vertices, sphere.GetVertexCount()}, {p_indices, sphere.GetIndexCount()});

	VulkanRHI::CommandBuffer cmd = swapchain.GetCurrentCommandBuffer();
	CHECK_VULKAN_RESULT(cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}));

	vk::BufferCopy regions_buf[32];
	auto           regions_count = 0;
	{
		auto add_region = [&](vk::BufferCopy const& region) { regions_buf[regions_count++] = region; };
		add_region({
			.srcOffset = vertices_offset,
			.dstOffset = vertices_offset,
			.size      = vertices_size_bytes,
		});
		add_region({
			.srcOffset = indices_offset,
			.dstOffset = indices_offset,
			.size      = indices_size_bytes,
		});
		for (auto const& ii : {
				 u32(BrdfFunctionType::eWeightsInBuffer),
				 u32(BrdfFunctionType::eWeightsInBufferF16),
				 u32(BrdfFunctionType::eCoopVec),
			 }) {
			add_region({
				.srcOffset = weights_offsets[ii],
				.dstOffset = weights_offsets[ii],
				.size      = networks[ii].GetParametersSize(),
			});
		}
		// Kan
		if (hasattr(&BRDFSample::kan_weights_file_name)) {
			add_region({
				.srcOffset = fastkan_offset,
				.dstOffset = fastkan_offset,
				.size      = kan.size_bytes() * 2,
			});
		}
	};
	std::span regions{regions_buf, static_cast<std::size_t>(regions_count)};

	cmd.copyBuffer(staging_buffer, device_buffer, std::size(regions), regions.data());

	// Cube
	if (hasattr(&BRDFSample::cubemap_folder_path)) {
		cmd.Barrier({
			.image         = cubemap_image,
			.aspectMask    = vk::ImageAspectFlagBits::eColor,
			.oldLayout     = vk::ImageLayout::eUndefined,
			.newLayout     = cubemap_image.SetLayout(vk::ImageLayout::eTransferDstOptimal).GetLayout(),
			.srcStageMask  = vk::PipelineStageFlagBits::eNone,
			.srcAccessMask = vk::AccessFlagBits::eNone,
			.dstStageMask  = vk::PipelineStageFlagBits::eTransfer,
			.dstAccessMask = vk::AccessFlagBits::eTransferWrite,
		});

		cmd.copyBufferToImage(
			staging_buffer,
			cubemap_image,
			vk::ImageLayout::eTransferDstOptimal,
			std::size(cube_regions),
			cube_regions);

		cmd.Barrier2({
			.image         = cubemap_image,
			.aspectMask    = vk::ImageAspectFlagBits::eColor,
			.oldLayout     = vk::ImageLayout::eUndefined,
			.newLayout     = cubemap_image.SetLayout(vk::ImageLayout::eShaderReadOnlyOptimal).GetLayout(),
			.srcStageMask  = vk::PipelineStageFlagBits2::eNone,
			.srcAccessMask = vk::AccessFlagBits2::eNone,
			.dstStageMask  = vk::PipelineStageFlagBits2::eAllGraphics,
			.dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead,
		});
	}

	CHECK_VULKAN_RESULT(cmd.end());
	CHECK_VULKAN_RESULT(queue.submit({{.commandBufferCount = 1, .pCommandBuffers = &cmd}}));
	CHECK_VULKAN_RESULT(queue.waitIdle());
}
