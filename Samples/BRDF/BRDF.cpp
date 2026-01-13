module;
#include <cassert> // assert

module BRDFSample;

// #include <cstddef>

#include "CheckResult.h"
#include "Shaders/BRDFConfig.h"

import NeuralGraphics;
import vulkan_hpp;
import WindowManager;
import VulkanExtensions;
import VulkanFunctions;
import Window;
import vk_mem_alloc;
import WeightsLoader;
import SamplesCommon;
import Math;
import std;

import FastKan;

import nlohmann.json;
using json = nlohmann::json;

using numeric::float16_t;

#ifdef COOPVEC_TYPE
#undef COOPVEC_TYPE
#endif
#define COOPVEC_TYPE numeric::float16_t

using namespace Utils;

using VulkanRHI::Buffer;
using VulkanRHI::Image;

static_assert(sizeof(BRDFConstants) <= 256);

void PrintMat4(float4x4 const& mat, int precision = 5, int digits = 2) {

	auto d = digits + precision + 1;

	auto print_elem = [&](float elem) { std::printf("%*.*f", d, precision, elem); };
	for (int i = 0; i < 4; ++i) {
		std::printf("%*.*f, %*.*f, %*.*f, %*.*f\n", d, precision, mat[i][0], d, precision, mat[i][1], d, precision, mat[i][2], d, precision, mat[i][3]);
	}
}

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
		.srcStride        = info.cols * GetVulkanComponentSize(info.src_component_type),
		.dstLayout        = info.dst_layout,
		.dstStride        = info.cols * GetVulkanComponentSize(info.dst_matrix_type),
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
		.srcStride        = linear.GetInputsCount() * GetVulkanComponentSize(src_component_type),
		.dstLayout        = dst_layout,
		.dstStride        = linear.GetInputsCount() * GetVulkanComponentSize(dst_matrix_type),
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

		std::size_t const src_weights_size_bytes = layer.GetWeightsCount() * GetVulkanComponentSize(src_component_type);
		std::size_t const src_biases_size_bytes  = layer.GetBiasesCount() * GetVulkanComponentSize(src_component_type);

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

void BRDFSample::CreateAndUploadBuffers(NetworkBufferInfo const& network_info) {
	sphere = UVSphere(1.0f, 32 * 2, 16 * 2);

	std::size_t vertices_size_bytes = sphere.GetVertexCount() * sizeof(UVSphere::Vertex);
	std::size_t indices_size_bytes  = sphere.GetIndexCount() * sizeof(UVSphere::IndexType);
	// std::size_t alignment             = kVectorAlignment;
	std::size_t vertices_size_aligned = AlignUpPowerOfTwo(vertices_size_bytes, kVectorAlignment);
	std::size_t indices_size_aligned  = AlignUpPowerOfTwo(indices_size_bytes, kVectorAlignment);

	auto const kan = *ReadKANWeights(kan_weights_file_name).or_else([](auto&& error) -> ReadKANResult {
		std::cerr << error << std::endl;
		std::exit(1);
		return {};
	});

	kan.repr();

	std::printf("Total size: %zu\n", kan.size());

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

	vk::DeviceSize const kMinBufferSize = 256 * 1024;

	std::size_t total_size_bytes =
		std::max(
			kMinBufferSize,
			vertices_size_aligned + indices_size_aligned
				+ networks[u32(BrdfFunctionType::eCoopVec)].GetParametersSize()
				+ networks[u32(BrdfFunctionType::eWeightsInBuffer)].GetParametersSize()
				+ networks[u32(BrdfFunctionType::eWeightsInBufferF16)].GetParametersSize()
				+ kan.size_bytes() * 5);

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

	// Update descriptor set
	{
		vk::DescriptorBufferInfo buffer_infos[] = {{.buffer = device_buffer, .offset = 0, .range = device_buffer.GetSize()}};

		vk::WriteDescriptorSet writes[] = {{
			.dstSet          = descriptor_set,
			.dstBinding      = 0,
			.dstArrayElement = 0,
			.descriptorCount = static_cast<u32>(std::size(buffer_infos)),
			.descriptorType  = vk::DescriptorType::eStorageBuffer,
			.pBufferInfo     = buffer_infos,
		}};

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

	std::printf("offset: %zu\n", offset);
	std::printf("weights_offsets: %zu\n", weights_offsets[u32(BrdfFunctionType::eWeightsInBufferF16)] + networks[u32(BrdfFunctionType::eWeightsInBufferF16)].GetParametersSize());

	auto dst_layout = LayoutTy::eRowMajor;
	kan_offsets     = write_fast_kan(device, kan, p_staging + offset, dst_layout, ComponentTy::eFloat32, ComponentTy::eFloat16);

	for (auto i = 0u; i < std::size(kan_offsets); ++i) {
		auto& offs = kan_offsets[i];
		offs.rbf_grid() += offset;
		offs.rbf_denom_inv() += offset;
		offs.spline_weight() += offset;
		offs.base_weight() += offset;
		offs.base_bias() += offset;
		std::printf("kan_offset[%u]:\n", i);
		std::printf("rbf_grid: %zu\n", offs.rbf_grid());
		std::printf("rbf_denom_inv: %zu\n", offs.rbf_denom_inv());
		std::printf("spline_weight: %zu\n", offs.spline_weight());
		std::printf("base_weight: %zu\n", offs.base_weight());
		std::printf("base_bias: %zu\n", offs.base_bias());
	}

	// std::printf("RBF_denom_inv: %f\n", static_cast<float>(*reinterpret_cast<float16_t* const>(p_staging + kan_offsets[0].rbf_denom_inv())));
	auto const src_val     = kan.layers()[0].base_bias().span(kan.buffer().data())[0];
	auto const staging_val = static_cast<float>(*reinterpret_cast<float16_t const*>(p_staging + kan_offsets[0].base_bias()));

	auto get_staging = [&](std::size_t offset) -> float { return static_cast<float>(*reinterpret_cast<float16_t const*>(p_staging + offset)); };

	auto verify_kan = [&](int layer_id, int buffer_id, int first_elem, int count) -> void {
		std::printf("verify_kan: %-20s\n", (kan.layers()[layer_id]).get_buffer_name(buffer_id).data());
		for (int i = first_elem; i < count; ++i) {
			auto const staging_val = get_staging((kan_offsets[layer_id]).get_buffer(buffer_id) + i * sizeof(float16_t));
			auto const src_val     = (kan.layers()[layer_id]).get_buffer(buffer_id).span(kan.buffer().data())[first_elem + i];
			std::printf("src_val: %f, staging_val: %f\n", src_val, staging_val);
		}
	};
	for (int i = 0; i < kan.layers().size(); ++i) {
		for (int j = 0; j < 5; ++j) {
			verify_kan(i, j, 0, 5);
		}
	}

	std::printf("src_val: %f, staging_val: %f\n", src_val, staging_val);

	if (verbose) {
		// networks[u32(BrdfFunctionType::eScalarBuffer)].Print();
		// networks[u32(BrdfFunctionType::eCoopVec)].Print();
		// PrintLayerBiases<float16_t>(networks[u32(BrdfFunctionType::eCoopVec)].GetLayer<Linear>(networks[u32(BrdfFunctionType::eCoopVec)].GetLayers().size() - 1), p_staging + this->optimal_weights_offset);
		// networks[u32(BrdfFunctionType::eScalarBuffer)].PrintLayerBiases(networks[u32(BrdfFunctionType::eScalarBuffer)].GetLayers().size() - 1, Component::eFloat32, p_staging + this->linear_weights_offset);
	}

	// DumpVertexData({p_vertices, sphere.GetVertexCount()}, {p_indices, sphere.GetIndexCount()});

	vk::CommandBuffer cmd = swapchain.GetCurrentCommandBuffer();
	CHECK_VULKAN_RESULT(cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}));
	vk::BufferCopy regions[] = {
		{
			.srcOffset = vertices_offset,
			.dstOffset = vertices_offset,
			.size      = vertices_size_bytes,
		},
		{
			.srcOffset = indices_offset,
			.dstOffset = indices_offset,
			.size      = indices_size_bytes,
		},
		// Linear
		{
			.srcOffset = weights_offsets[u32(BrdfFunctionType::eWeightsInBuffer)],
			.dstOffset = weights_offsets[u32(BrdfFunctionType::eWeightsInBuffer)],
			.size      = networks[u32(BrdfFunctionType::eWeightsInBuffer)].GetParametersSize(),
		},
		// Linear f16
		{
			.srcOffset = weights_offsets[u32(BrdfFunctionType::eWeightsInBufferF16)],
			.dstOffset = weights_offsets[u32(BrdfFunctionType::eWeightsInBufferF16)],
			.size      = networks[u32(BrdfFunctionType::eWeightsInBufferF16)].GetParametersSize(),
		},
		// Optimal
		{
			.srcOffset = weights_offsets[u32(BrdfFunctionType::eCoopVec)],
			.dstOffset = weights_offsets[u32(BrdfFunctionType::eCoopVec)],
			.size      = networks[u32(BrdfFunctionType::eCoopVec)].GetParametersSize(),
		},
		// Kan
		// {
		// 	.srcOffset = weights_offsets[u32(BrdfFunctionType::eKan)],
		// 	.dstOffset = weights_offsets[u32(BrdfFunctionType::eKan)],
		// 	.size      = networks[u32(BrdfFunctionType::eKan)].GetParametersSize(),
		// },
	};

	cmd.copyBuffer(staging_buffer, device_buffer, std::size(regions), regions);

	CHECK_VULKAN_RESULT(cmd.end());
	CHECK_VULKAN_RESULT(queue.submit({{.commandBufferCount = 1, .pCommandBuffers = &cmd}}));
	CHECK_VULKAN_RESULT(queue.waitIdle());
}

auto BRDFSample::GetQueryResult() -> u64 {
	vk::Result result =
		device.getQueryPoolResults(
			timestamp_query_pool,
			GetCurrentTimestampIndex(),
			kTimestampsPerFrame,
			sizeof(timestamp_results[0]) * kTimestampsPerFrame,
			GetCurrentTimestampResult(),
			sizeof(u64),
			vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);

	switch (result) {
	case vk::Result::eSuccess: {
		u64* current_timestamps = GetCurrentTimestampResult();
		u64  start              = current_timestamps[0];
		u64  end                = current_timestamps[1];
		u64  elapsed            = end > start ? end - start : 0;
		return elapsed;
		break;
	}
	case vk::Result::eNotReady:
		// timestamp_valid = false;
		std::printf("Timestamp not ready\n");
		break;
	default:
		CHECK_VULKAN_RESULT(result);
	}
	return 0ull;
}

auto BRDFSample::DrawWindow() -> u64 {
	if (function_id)
		return DrawWindow(pipelines_header[u32(*function_id)]);
	else
		return DrawWindow(pipelines[u32(function_type)]);
};

auto BRDFSample::DrawWindow(vk::Pipeline pipeline) -> u64 {
	auto HandleSwapchainResult = [this](vk::Result result) -> bool {
		switch (result) {
		case vk::Result::eSuccess:           return true;
		case vk::Result::eErrorOutOfDateKHR: swapchain_dirty = true; return false;
		case vk::Result::eSuboptimalKHR:     swapchain_dirty = true; return true;
		default:
			CHECK_VULKAN_RESULT(result);
		}
		return false;
	};
	CHECK_VULKAN_RESULT(device.waitForFences(1, &swapchain.GetCurrentFence(), vk::True, std::numeric_limits<u32>::max()));
	CHECK_VULKAN_RESULT(device.resetCommandPool(swapchain.GetCurrentCommandPool()));
	if (!HandleSwapchainResult(swapchain.AcquireNextImage())) return 0ull;
	CHECK_VULKAN_RESULT(device.resetFences(1, &swapchain.GetCurrentFence()));
	RecordCommands(pipeline);
	if (!HandleSwapchainResult(swapchain.SubmitAndPresent(queue, queue))) return 0ull;

	u64 elapsed = GetQueryResult();
	swapchain.EndFrame();
	return elapsed;
}

void BRDFSample::RecordCommands(vk::Pipeline pipeline) {
	int x, y, width, height;
	window.GetRect(x, y, width, height);

	auto depth_extent = depth_image.GetExtent();
	if (static_cast<u32>(width) > depth_extent.width || static_cast<u32>(height) > depth_extent.height) {
		depth_image.Recreate({static_cast<u32>(width), static_cast<u32>(height), 1});
	}

	vk::Rect2D               render_rect{{0, 0}, {static_cast<u32>(width), static_cast<u32>(height)}};
	VulkanRHI::CommandBuffer cmd = swapchain.GetCurrentCommandBuffer();
	CHECK_VULKAN_RESULT(cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}));
	cmd.resetQueryPool(timestamp_query_pool, GetCurrentTimestampIndex(), kTimestampsPerFrame);
	cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, timestamp_query_pool, GetCurrentTimestampIndex());

	vk::Image swapchain_image = swapchain.GetCurrentImage();
	// cmd.SetViewport({0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f});
	cmd.SetViewport({0.0f, static_cast<float>(height), static_cast<float>(width), -static_cast<float>(height), 0.0f, 1.0f});
	cmd.SetScissor(render_rect);
	cmd.Barrier({
		.image         = swapchain_image,
		.aspectMask    = vk::ImageAspectFlagBits::eColor,
		.oldLayout     = vk::ImageLayout::eUndefined,
		.newLayout     = vk::ImageLayout::eColorAttachmentOptimal,
		.srcStageMask  = vk::PipelineStageFlagBits2::eNone,
		.srcAccessMask = vk::AccessFlagBits2::eNone,
		.dstStageMask  = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
	});
	cmd.BeginRendering({
		.renderArea       = render_rect,
		.colorAttachments = {{{
			.imageView   = swapchain.GetCurrentImageView(),
			.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
			.loadOp      = vk::AttachmentLoadOp::eClear,
			.storeOp     = vk::AttachmentStoreOp::eStore,
			.clearValue  = {{{{0.5f, 0.5f, 0.5f, 0.0f}}}},
			// .clearValue = {{{{1.f, 1.f, 1.f, 1.0f}}}},
			// .clearValue  = {{{{0.f, 0.f, 0.f, 1.0f}}}},
		}}},
		.depthAttachment  = {
			 .imageView   = depth_image.GetView(),
			 .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
			 .loadOp      = vk::AttachmentLoadOp::eClear,
			 .storeOp     = vk::AttachmentStoreOp::eDontCare,
			 .clearValue  = {{{{1.0f, 0}}}},
        },
	});
	// auto pipeline = pipelines[u32(function_type)];
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

	camera.getForward() *= -1.0;
	camera.updateProjectionViewInverse();
	camera.getForward() *= -1.0;
	BRDFConstants constants{
		.view_proj = camera.getProjViewInv(),
		.material  = {
			 .base_color = float4{0.1, 0.6, 0.8, 1.0} * 0.2,
			 .metallic   = 0.0f,
			 .roughness  = 0.5f,
        },
		.light = {
			.position          = vec3(1.2, 1.2, 1.2),
			.range             = 10.0,
			.color             = vec3(0.75, 0.75, 0.75),
			.intensity         = 8.0,
			.ambient_color     = vec3(0.9, 0.9, 0.9),
			.ambient_intensity = 0.03,
		},
		.camera_pos = camera.getPosition(),
	};

	// PrintMat4(camera.getView());
	// std::printf("\n");

	// Update weight offsets in push constants
	auto network = networks[u32(function_type)];
	for (int i = 0; i < network.GetLayers().size(); ++i) {
		auto layer       = network.GetLayer<Linear>(i);
		auto offset_base = weights_offsets[u32(function_type)];

		// ENABLE_MLP
		// constants.weights_offsets[i] = offset_base + layer.GetWeightsOffset();
		// constants.bias_offsets[i]    = offset_base + layer.GetBiasesOffset();

		// std::printf("Layer %d weights_offset %d bias_offset %d\n", i, constants.weights_offsets[i], constants.bias_offsets[i]);
	}

	u32 const constants_offset = 0u;
	cmd.pushConstants(
		pipeline_layout,
		vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
		constants_offset, sizeof(constants), &constants);

	// std::printf("camera pos %f %f %f\n", constants.camera_pos.x, constants.camera_pos.y, constants.camera_pos.z);

	// u32 vertex_count = GetCubeVertices().size();
	u32 vertex_count = sphere.GetVertexCount();
	cmd.bindVertexBuffers(0, device_buffer, vertices_offset);
	cmd.bindIndexBuffer(device_buffer, indices_offset, vk::IndexType::eUint32);
	// cmd.draw(3, 1, 0, 0);
	// cmd.draw(vertex_count, 1, 0, 0);
	cmd.drawIndexed(sphere.GetIndexCount(), 1, 0, 0, 0);
	cmd.endRendering();
	cmd.Barrier({
		.image         = swapchain_image,
		.aspectMask    = vk::ImageAspectFlagBits::eColor,
		.oldLayout     = vk::ImageLayout::eColorAttachmentOptimal,
		.newLayout     = vk::ImageLayout::ePresentSrcKHR,
		.srcStageMask  = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
		.dstStageMask  = vk::PipelineStageFlagBits2::eNone,
		.dstAccessMask = vk::AccessFlagBits2::eNone,
	});

	// Write timestamp at end
	cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, timestamp_query_pool, GetCurrentTimestampIndex() + 1);
	CHECK_VULKAN_RESULT(cmd.end());
}

void BRDFSample::RecreateSwapchain(int width, int height) {
	for (auto& frame : swapchain.GetFrameData()) {
		CHECK_VULKAN_RESULT(device.waitForFences(1, &frame.GetFence(), vk::True, std::numeric_limits<u32>::max()));
	}
	CHECK_VULKAN_RESULT(swapchain.Recreate(width, height));
	swapchain_dirty = false;
}

void BRDFSample::SaveSwapchainImageToFile(std::string_view filename) {
	CHECK_VULKAN_RESULT(device.waitForFences(1, &swapchain.GetCurrentFence(), vk::True, std::numeric_limits<u32>::max()));

	int x, y, width, height;
	window.GetRect(x, y, width, height);
	auto const image             = swapchain.GetCurrentImage();
	auto const image_view        = swapchain.GetCurrentImageView();
	auto const new_layout        = vk::ImageLayout::eTransferSrcOptimal;
	auto const image_aspect      = vk::ImageAspectFlagBits::eColor;
	auto const image_subresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);

	auto extent            = swapchain.GetExtent();
	auto format_block_size = vk::blockSize(swapchain.GetFormat());

	vk::DeviceSize const image_size = extent.width * extent.height * format_block_size;

	if (image_size >= staging_buffer.GetSize()) {
		staging_buffer.Destroy();
		// clang-format off
		CHECK_VULKAN_RESULT(staging_buffer.Create(device, vma_allocator, {
			.size   = image_size,
			.usage  = vk::BufferUsageFlagBits::eTransferSrc,
			.memory = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		}));
		// clang-format on
	}

	VulkanRHI::CommandBuffer cmd = swapchain.GetCurrentCommandBuffer();
	CHECK_VULKAN_RESULT(cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}));

	cmd.Barrier({
		.image         = image,
		.oldLayout     = vk::ImageLayout::ePresentSrcKHR,
		.newLayout     = new_layout,
		.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
		.srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
		.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer,
		.dstAccessMask = vk::AccessFlagBits2::eTransferRead,
	});

	auto region = vk::BufferImageCopy{
		.bufferOffset      = 0,
		.bufferRowLength   = 0,
		.bufferImageHeight = 0,
		.imageSubresource  = image_subresource,
		.imageOffset       = vk::Offset3D{0, 0, 0},
		.imageExtent       = vk::Extent3D{extent.width, extent.height, 1},
	};

	cmd.copyImageToBuffer(image, new_layout, staging_buffer, 1, &region);

	auto cmd_info = vk::CommandBufferSubmitInfo{.commandBuffer = cmd};
	CHECK_VULKAN_RESULT(cmd.end());
	CHECK_VULKAN_RESULT(queue.submit2(vk::SubmitInfo2{
		.commandBufferInfoCount = 1,
		.pCommandBufferInfos    = &cmd_info,
	}))
	CHECK_VULKAN_RESULT(queue.waitIdle());

	auto data   = staging_buffer.GetMappedData();
	auto stride = extent.width * format_block_size;
	if (stbi_write_bmp(filename.data(), extent.width, extent.height, format_block_size, data)) {
		// if (stbi_write_png(filename.data(), extent.width, extent.height, format_block_size, data, 0)) {
		std::printf("Saved %s\n", filename.data());
	} else {
		std::printf("Failed to write %s\n", filename.data());
	}
	// pending_image_save = false;
}

void BRDFSample::Run() {
	do {
		WindowManager::WaitEvents();
		if (window.GetShouldClose()) break;
		int x, y, width, height;
		window.GetRect(x, y, width, height);
		if (width <= 0 || height <= 0) continue;
		u64       elapsed_ns = DrawWindow();
		verbose&& std::printf("%f ms\n", elapsed_ns / 1000000.0);
	} while (true);
}

template <typename Range, typename Proj = std::identity>
constexpr inline auto contains(Range&& range, auto&& value, Proj&& proj = std::identity{}) {
	for (auto&& v : range)
		if (std::invoke(proj, v) == value)
			return true;
	return false;
};

void BRDFSample::RunBenchmark(TestOptions const& options) {
	struct TestData {
		vk::Pipeline                pipeline;
		VulkanCoopVecNetwork const* network;
	};
	// WindowManager::WaitEvents();
	// if (window.GetShouldClose()) return;

	// window.SetWindowMode(WindowMode::eFullscreen);
	auto [width, height] = options.resolution;
	window.SetSize(width, height);
	// window.Hide();
	// std::printf("Resizing to %dx%d\n", width, height);
	RecreateSwapchain(width, height);
	constexpr u32 kTestRunsCount = 64;

	// constexpr u32 kMaxTestKinds = std::to_underlying(BrdfFunctionType::eCount);
	constexpr u32 kMaxTestKinds = kTestFunctionsCount;

	constexpr u32 kMaxTests = 64;
	// std::vector<std::array<u64, kMaxTestKinds>> test_times(kTestRunsCount);
	std::array<std::array<u64, kMaxTests>, kTestRunsCount> test_times;

	int first_test{}, last_test = kMaxTestKinds;

	if (benchmark_single) {
		first_test = std::to_underlying(function_type);
		last_test  = first_test + 1;
	} else {
		first_test = *function_id;
		last_test  = first_test + 1;
	}

	bool is_header = true;
	// bool is_header = false;
	if (is_header) {
		first_test = 0;
		last_test  = kTestFunctionsCount;
		// last_test  = 5;
	}

	// std::printf("Running %d tests\n", last_test - first_test);
	// std::printf("test id: %d\n", first_test);

	// BrdfFunctionType skip[] = {BrdfFunctionType::eWeightsInHeader};
	BrdfFunctionType skip[] = {};

	// std::mem_fn(&BRDFSample::DrawWindow);

	auto draw = [&](u32 id) {
		if (is_header) {
			return DrawWindow(pipelines_header[id]);
		} else {
			return DrawWindow(pipelines[id]);
		};
	};

	// Warm up gpu clocks
	constexpr u32 kWarmupCount = 2;
	for (u32 t_i = first_test; t_i < last_test; ++t_i) {
		if (contains(skip, BrdfFunctionType(t_i))) continue;
		for (u32 iter = 0; iter < kWarmupCount; ++iter) {
			(void)draw(t_i);
		}
	}

	bool with_in_header = false;

	for (u32 t_i = first_test; t_i < last_test; ++t_i) {
		if (contains(skip, BrdfFunctionType(t_i))) continue;
		for (u32 iter = 0; iter < kTestRunsCount; ++iter) {
			// WindowManager::PollEvents();
			u64   time_nanoseconds = draw(t_i);
			float ns_per_tick      = physical_device.GetNsPerTick();
			float elapsed_ms       = (time_nanoseconds * ns_per_tick) / 1e6f;
			test_times[iter][t_i]  = time_nanoseconds;
		}
	}

	char const* names[] = {"Classic", "CoopVec", "WeightsInBuffer", "WeightsInBufferFloat16", "WeightsInHeader", "Kan"};

	// Print csv
	// std::printf("Print csv\n");
	// std::printf("Classic,CoopVec,WeightsInBuffer,WeightsInBufferFloat16,WeightsInHeader\n");

	char const* header_names[] = {
#define BRDF_NAME(x) #x,
// #include "SINEKAN_HeaderNames.def"
// #include "FASTKAN_HeaderNames.def"
#include "CHEBYKAN_HeaderNames.def"
		// #include "RELUKAN_HeaderNames.def"
	};

	for (u32 t_i = first_test; t_i < last_test; ++t_i) {
		if (contains(skip, BrdfFunctionType(t_i))) continue;
		if (is_header) {
			std::printf("%s", header_names[t_i]);
		} else {
			// std::printf("t_i %u", t_i);
			std::printf("%s", names[t_i]);
		}
		if (t_i < last_test - 1 && !contains(skip, BrdfFunctionType(t_i + 1))) std::printf(",");
	}
	std::printf("\n");

	// std::printf("Print times\n");
	for (u32 iter = 0; iter < kTestRunsCount; ++iter) {
		auto const& tests_row = test_times[iter];
		// print with ,
		for (u32 t_i = first_test; t_i < last_test; ++t_i) {
			if (contains(skip, BrdfFunctionType(t_i))) continue;
			std::printf("%llu", tests_row[t_i]);
			if (t_i < last_test - 1 && !contains(skip, BrdfFunctionType(t_i + 1))) std::printf(",");
		}
		std::printf("\n");
	}
}

auto BRDFSample::ParseArgs(int argc, char const* argv[]) -> char const* {
	auto args_range = std::span(argv + 1, argc - 1);

	if (std::ranges::contains(args_range, std::string_view("--help")))
		return "--help";

	for (auto it = args_range.begin(); it != args_range.end(); ++it) {
		auto arg = std::string_view(*it);
		if (arg == "--benchmark" || arg == "-b") is_test_mode = true;
		else if (arg == "--verbose" || arg == "-v") verbose = true;
		else if (arg == "--validation") use_validation = true;
		else if (arg == "--kind") {
			if ((it + 1) == args_range.end()) return "expected <kind>";
			auto kind = std::string_view(*(it + 1));
			int  value;
			if (std::from_chars(kind.data(), kind.data() + kind.size(), value).ec != std::errc()) return *(it + 1);
			if (value < 0 || value >= std::to_underlying(BrdfFunctionType::eCount)) return *(it + 1);
			function_type    = static_cast<BrdfFunctionType>(value);
			benchmark_single = true;
			++it;
		} else if (arg == "-f") {
			if ((it + 1) == args_range.end()) return "expected <id>";
			auto str = std::string_view(*(it + 1));
			int  value;
			if (std::from_chars(str.data(), str.data() + str.size(), value).ec != std::errc()) return *(it + 1);
			if (value < 0 || value >= kTestFunctionsCount) return *(it + 1);
			// function_type    = BrdfFunctionType::eWeightsInHeader;
			benchmark_single = true;
			function_id      = value;
			++it;
		} else if (arg == "-w") { // input weights file
			if ((it + 1) == args_range.end()) return "expected <file>";
			auto str = std::string_view(*(it + 1));
			// if (!std::filesystem::exists(str)) return *(it + 1);
			weights_file_name = str;
			++it;
		} else if (arg == "-kw") { // input weights file
			if ((it + 1) == args_range.end()) return "expected <file>";
			auto str = std::string_view(*(it + 1));

			kan_weights_file_name = str;
			++it;
		} else return *it;
	}

	return nullptr;
}

auto PrintUsage([[maybe_unused]] int argc, char const* argv[]) -> void {
	std::printf("Usage: %s [--help] [--benchmark | -b] [--verbose | -v] [--validation] [--kind <kind>]\n",
				std::filesystem::path(argv[0]).filename().string().c_str());
	std::printf("  --kind <kind>\n");
	std::printf("      Kind of BRDF function to run:\n");
	std::printf("        0: Classic\n");
	std::printf("        1: Coop Vec (Default)\n");
	std::printf("        2: Weights in buffer\n");
	std::printf("        3: Weights in buffer float16\n");
	std::printf("        4: Weights in header\n");
	std::printf("        5: Kan\n");
	std::printf("  --benchmark | -b\n");
	std::printf("      Run benchmark\n");
	std::printf("  --verbose | -v\n");
	std::printf("      Run benchmark with verbose output\n");
	std::printf("  --validation\n");
	std::printf("      Enable validation\n");
	std::printf("\n");
};

auto main(int argc, char const* argv[]) -> int {
	std::filesystem::current_path(std::filesystem::absolute(argv[0]).parent_path());
	BRDFSample sample;

	if (char const* unknown_arg = sample.ParseArgs(argc, argv); unknown_arg) {
		if (unknown_arg != std::string_view("--help"))
			std::printf("Error in argument: %s\n", unknown_arg);
		PrintUsage(argc, argv);
		return 0;
	}

	TestOptions options{
		.resolution = {640, 480},
		.test_count = 64,
	};

	int2 res_arr[] = {
		{1920, 1080},
		// {3840, 2160},
		// {512, 512},
		{640, 480},
		{1280, 720},
		{1920, 1080},
		{2560, 1440},
		// {3840, 2160},
	};

	auto res_count = //

		// std::size(res_arr);
		1;

	sample.Init();
	if (sample.IsTestMode()) {
		for (int i = 0; i < res_count; ++i) {
			options.resolution = res_arr[i];
			// std::printf("resolution: %d x %d\n", res_arr[i].x, res_arr[i].y);
			sample.RunBenchmark(options);
		}
	} else {
		sample.Run();
	}
	return 0;
}
