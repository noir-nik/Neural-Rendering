module;
#include "CheckResult.h"
#include "Log.h"
#include "Shaders/BRDFBindings.h"
#include <cassert> // assert

module BRDFSample;
import NeuralGraphics;
import vulkan;
import WeightsLoader;
import SamplesCommon;
import Math;
import std;
import FastKANCoopVec;

import nlohmann.json;
using json = nlohmann::json;
using namespace mesh;

using numeric::float16_t;

template <typename T>
using CSpan = std::span<T const>;
template <typename T>
using Span = std::span<T>;

void DumpVertexData(std::span<const Vertex> vertices, std::span<const UVSphere::IndexType> indices) {
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

using ComponentTy = vk::ComponentTypeKHR;
using LayoutTy    = vk::CooperativeVectorMatrixLayoutNV;

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

void BRDFSample::CreateAndUploadBuffers(NetworkBufferInfo const& network_info) {
	LOG_DEBUG("BRDFSample::CreateAndUploadBuffers()");

	bool is_obj = hasattr(&BRDFSample::obj_path);
	// if (!stat(obj_path.data(), &network_info.file_stat)) is_sphere = true;

	std::vector<Vertex>              vertices;
	std::vector<UVSphere::IndexType> indices;
	if (is_obj) {

		auto const& [vv, ii]        = LoadObj(obj_path);
		std::tie(vertices, indices) = std::pair{std::move(vv), std::move(ii)};

		// this->num_vertices          = vertices.size();
		// this->num_indices           = indices.size();
		std::tie(this->num_vertices, this->num_indices) = std::pair{vertices.size(), indices.size()};
		// std::tie(this->num_vertices, this->num_indices) = std::views::transform(std::tuple{vertices, indices}, [](auto const& i) { return i.size(); });

		// vertices                    = std::move(vv);
		// indices                     = std::move(ii);
		// std::tie(vertices, indices) = [this] {
		// 	auto const& [vv, ii] = LoadObj(obj_path);
		// 	return std::tuple{std::move(vv), std::move(ii)};
		// }();
		this->vv = vertices[0];
	} else {
		sphere             = UVSphere(1.0f, 32 * 2, 16 * 2);
		this->num_vertices = sphere.GetVertexCount();
		this->num_indices  = sphere.GetIndexCount();
	}

	std::size_t vertices_size_bytes = AlignUpPowerOfTwo(std::size_t{this->num_vertices * sizeof(Vertex)}, kVectorAlignment);
	std::size_t indices_size_bytes  = AlignUpPowerOfTwo(std::size_t{this->num_indices * sizeof(UVSphere::IndexType)}, kVectorAlignment);

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

	if (0)
		if (layers.size() != 5) {
			std::printf("Error loading weights : wrong number of layers\n");
			std::exit(1);
		}

	// Get total size for buffer
	if (with_coop_vec()) {
		networks[u32(BrdfFunctionType::eWeightsInBuffer)].Init(layers);
		networks[u32(BrdfFunctionType::eWeightsInBufferF16)].Init(layers);
		networks[u32(BrdfFunctionType::eCoopVec)].Init(layers);
		CHECK_VULKAN_RESULT(networks[u32(BrdfFunctionType::eWeightsInBuffer)].UpdateOffsetsAndSize(device, LayoutTy::eRowMajor, ComponentTy::eFloat32, ComponentTy::eFloat32));
		CHECK_VULKAN_RESULT(networks[u32(BrdfFunctionType::eWeightsInBufferF16)].UpdateOffsetsAndSize(device, LayoutTy::eRowMajor, ComponentTy::eFloat16, ComponentTy::eFloat16));
		CHECK_VULKAN_RESULT(networks[u32(BrdfFunctionType::eCoopVec)].UpdateOffsetsAndSize(device, LayoutTy::eInferencingOptimal, ComponentTy::eFloat16, ComponentTy::eFloat16));
	}

	CubeMetadata         cube_data;
	HdriToCubemap<float> hdri_cube;

	// if (std::filesystem::exists(cubemap_folder_path)){

	// }

	auto cubemap_found = true;

	bool const is_hdr_cubemap = [&] {
		auto cube_path = std::filesystem::path(cubemap_folder_path);
		if (std::filesystem::is_directory(cube_path)) {
			return false;
		} else if (cube_path.extension() == ".hdr") {
			return true;
		} else {
			cubemap_found = false;
			return false;
		}
	}();

	if (!cubemap_found) {
		LOG_WARN("%s\n", "Valid cubemap not found");
	}

	auto const get_extent = [](auto cube) {
		return vk::Extent3D{static_cast<u32>(cube.get_resolution()), static_cast<u32>(cube.get_resolution()), 1};
	};

	if (cubemap_found) {
		if (is_hdr_cubemap) {
			auto hdri_res = 1024;
			hdri_cube.init(cubemap_folder_path, hdri_res);

			std::printf("res: %d\n", hdri_cube.get_resolution());
			std::printf("channels: %d\n", hdri_cube.get_num_channels());
			assert(hdri_cube.image_size_bytes() == hdri_res * hdri_res * sizeof(float) * hdri_cube.get_num_channels());

		} else {
			cube_data.init(cubemap_folder_path);
		}
	}
	auto cube_extent = is_hdr_cubemap ? get_extent(hdri_cube) : get_extent(cube_data);

	if (hasattr(&BRDFSample::cubemap_folder_path)) {

		auto cube_format =
			is_hdr_cubemap
				? (hdri_cube.get_num_channels() == 4
					   ? vk::Format::eR32G32B32A32Sfloat
					   : vk::Format::eR32G32B32Sfloat)
				: vk::Format::eR8G8B8A8Unorm;

		cubemap_image.Create(
			device, vma_allocator, allocator,
			{.image_info = {
				 .flags         = vk::ImageCreateFlagBits::eCubeCompatible,
				 .imageType     = vk::ImageType::e2D,
				 .format        = cube_format,
				 .extent        = cube_extent,
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
			vertices_size_bytes + indices_size_bytes
				+ (with_coop_vec()
					   ? networks[u32(BrdfFunctionType::eCoopVec)].GetParametersSize()
							 + networks[u32(BrdfFunctionType::eWeightsInBuffer)].GetParametersSize()
							 + networks[u32(BrdfFunctionType::eWeightsInBufferF16)].GetParametersSize()
					   : 0)
				+ kan.size_bytes() * 5
				+ cube_data.size_bytes() * 2 * cube_data.is_valid()
				+ hdri_cube.size_bytes() * 2 * hdri_cube.is_valid());

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
		.flags        = {},
		.magFilter    = vk::Filter::eLinear,
		.minFilter    = vk::Filter::eLinear,
		.addressModeU = vk::SamplerAddressMode::eClampToEdge,
		.addressModeV = vk::SamplerAddressMode::eClampToEdge,
		.addressModeW = vk::SamplerAddressMode::eClampToEdge,
		.mipLodBias   = 0.0f,
		// .anisotropyEnable = vk::True,
		.maxAnisotropy = 16.0f,
		.compareEnable = vk::False,
		.compareOp     = vk::CompareOp::eNever,
		.minLod        = 0.0f,
		.maxLod        = 1.0f,
		.borderColor   = vk::BorderColor::eFloatOpaqueWhite,
	};

	CHECK_VULKAN_RESULT(device.createSampler(&sampler_info, GetAllocator(), &cubemap_sampler));

	// Update descriptor set
	{
		vk::DescriptorBufferInfo buffer_infos[]  = {{.buffer = device_buffer, .offset = 0, .range = device_buffer.GetSize()}};
		vk::DescriptorImageInfo  texture_infos[] = {{
			.sampler     = cubemap_sampler,
			.imageView   = cubemap_image.GetView(),
			.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
		}};
		vk::DescriptorImageInfo  image_infos[]   = {{
			.sampler     = cubemap_sampler,
			.imageView   = accumulator_image.GetView(),
			.imageLayout = vk::ImageLayout::eGeneral,
		}};

		auto make_write =
			[&](uint32_t                        dstBinding,
				uint32_t                        descriptorCount,
				vk::DescriptorType              descriptorType,
				vk::DescriptorBufferInfo const* pBufferInfo,
				vk::DescriptorImageInfo const*  pImageInfo) {
				return vk::WriteDescriptorSet{
					.dstSet          = descriptor_set,
					.dstBinding      = dstBinding,
					.dstArrayElement = 0,
					.descriptorCount = descriptorCount,
					.descriptorType  = descriptorType,
					.pImageInfo      = pImageInfo,
					.pBufferInfo     = pBufferInfo,
				};
			};

		using enum vk::DescriptorType;

		auto const writes = std::array{
			make_write(BINDING_STORAGE_BUFFER, std::size(buffer_infos), eStorageBuffer, buffer_infos, {}),
			make_write(BINDING_TEXTURE, std::size(texture_infos), eCombinedImageSampler, {}, texture_infos),
			make_write(BINDING_STORAGE_IMAGE, std::size(image_infos), eStorageImage, {}, image_infos),
		};

		auto descriptor_copy_count = 0u;
		auto copy_descriptor_sets  = nullptr;
		device.updateDescriptorSets(std::size(writes), std::data(writes), descriptor_copy_count, copy_descriptor_sets);
	}

	std::size_t offset    = 0;
	this->vertices_offset = offset;
	offset += vertices_size_bytes;
	this->indices_offset = offset;
	offset += indices_size_bytes;

	for (auto i = 1u; i < std::size(networks); ++i) {
		offset             = AlignUpPowerOfTwo(offset, CoopVecUtils::GetMatrixAlignment());
		weights_offsets[i] = offset;
		offset += AlignUpPowerOfTwo(networks[i].GetParametersSize(), CoopVecUtils::GetMatrixAlignment());
	}

	auto p_staging  = static_cast<std::byte*>(staging_buffer.GetMappedData());
	auto p_vertices = reinterpret_cast<Vertex*>(p_staging + this->vertices_offset);
	auto p_indices  = reinterpret_cast<UVSphere::IndexType*>(p_staging + this->indices_offset);

	if (is_obj) {

		std::memcpy(p_vertices, vertices.data(), Span(vertices).size_bytes());
		std::memcpy(p_indices, indices.data(), Span(indices).size_bytes());
	} else {
		sphere.WriteVertices(p_vertices);
		sphere.WriteIndices(p_indices);
	}
	auto brdf_weights_src = std::span{reinterpret_cast<std::byte*>(brdf_weights_vec.data()), brdf_weights_vec.size() * sizeof(float)};
	if (with_coop_vec()) {
		write_network<float, numeric::float16_t>(device, networks[u32(BrdfFunctionType::eCoopVec)], brdf_weights_src.data(), p_staging + weights_offsets[u32(BrdfFunctionType::eCoopVec)], LayoutTy::eInferencingOptimal, ComponentTy::eFloat32, ComponentTy::eFloat16);
		write_network<float, float>(device, networks[u32(BrdfFunctionType::eWeightsInBuffer)], brdf_weights_src.data(), p_staging + weights_offsets[u32(BrdfFunctionType::eWeightsInBuffer)], LayoutTy::eRowMajor, ComponentTy::eFloat32, ComponentTy::eFloat32);
		write_network<float, numeric::float16_t>(device, networks[u32(BrdfFunctionType::eWeightsInBufferF16)], brdf_weights_src.data(), p_staging + weights_offsets[u32(BrdfFunctionType::eWeightsInBufferF16)], LayoutTy::eRowMajor, ComponentTy::eFloat32, ComponentTy::eFloat16);
	}
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

	constexpr auto num_cube_sides = 6;

	// u64 cube_offsets[num_cube_sides]{};

	// vk::BufferImageCopy cube_region{};
	auto cube_regions = std::array<vk::BufferImageCopy, num_cube_sides>{};

	bool is_cube_copied = false;

	auto const copy_cube = [&](auto in_cube) {
		for (u32 const side_id : Utils::indices(num_cube_sides)) {
			auto imdata = in_cube.get_faces()[side_id];

			auto const sbytes = in_cube.image_size_bytes();
			std::memcpy(p_staging + offset, imdata, sbytes);
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
				.imageExtent = get_extent(in_cube),
			};
			offset = AlignUpPowerOfTwo(offset + sbytes, kVectorAlignment);
		}
	};

	if (cube_data.is_valid()) {
		copy_cube(cube_data);
		is_cube_copied = true;
		cube_data.destroy();
	} else if (hdri_cube.is_valid()) {
		copy_cube(hdri_cube);
		is_cube_copied = true;
		hdri_cube.destroy();
	}

	// auto cubes = std::tuple{cube_data, hdri_cube};

	// std::apply([](auto&& cube) {}, cubes);

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

		if (with_coop_vec()) {
			for (auto const& ii : {u32(BrdfFunctionType::eWeightsInBuffer), u32(BrdfFunctionType::eWeightsInBufferF16), u32(BrdfFunctionType::eCoopVec)}) {
				if (!with_coop_vec() && ii == u32(BrdfFunctionType::eCoopVec)) continue;
				add_region({
					.srcOffset = weights_offsets[ii],
					.dstOffset = weights_offsets[ii],
					.size      = networks[ii].GetParametersSize(),
				});
			}
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
	if (is_cube_copied) {
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
			std::data(cube_regions));

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
		this->is_cubemap_loaded_ = true;
	}

	cmd.Barrier({
		.image         = accumulator_image,
		.aspectMask    = vk::ImageAspectFlagBits::eColor,
		.oldLayout     = vk::ImageLayout::eUndefined,
		.newLayout     = accumulator_image.SetLayout(vk::ImageLayout::eGeneral).GetLayout(),
		.srcStageMask  = vk::PipelineStageFlagBits::eNone,
		.srcAccessMask = vk::AccessFlagBits::eNone,
		.dstStageMask  = vk::PipelineStageFlagBits::eTransfer,
		.dstAccessMask = vk::AccessFlagBits::eNone,
	});

	// depth
	cmd.Barrier({
		.image         = depth_image,
		.aspectMask    = vk::ImageAspectFlagBits::eDepth,
		.oldLayout     = vk::ImageLayout::eUndefined,
		.newLayout     = depth_image.SetLayout(vk::ImageLayout::eDepthAttachmentOptimal).GetLayout(),
		.srcStageMask  = vk::PipelineStageFlagBits::eNone,
		.srcAccessMask = vk::AccessFlagBits::eNone,
		.dstStageMask  = vk::PipelineStageFlagBits::eEarlyFragmentTests,
		.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite,
	});

	CHECK_VULKAN_RESULT(cmd.end());
	CHECK_VULKAN_RESULT(queue.submit({{.commandBufferCount = 1, .pCommandBuffers = &cmd}}));
	CHECK_VULKAN_RESULT(queue.waitIdle());
}
