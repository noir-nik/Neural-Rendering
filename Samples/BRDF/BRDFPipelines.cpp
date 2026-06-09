module;

#include "stddef.h" // offsetof
// #include <cstdio>   // stdin
#include "CheckResult.h"
#include <cassert>

module BRDFSample;

namespace fs = std::filesystem;

#define LF(fn) [&](auto&&... args) { return fn; }
#define LF_PACK(fn) [&](auto&&... args) { (fn, ...); }

auto error_read_file(std::string_view name) -> Utils::BinDataOption {
	std::printf("Failed to read shader file: %*s\n", int(std::size(name)), std::data(name));
	std::exit(1);
	return {};
};

auto readfile(SV fpath) -> Utils::BinDataOption {
	// return Utils::ReadBinaryFile(fpath).or_else(LF(error_read_file(fpath)));
	return Utils::ReadBinaryFile(fpath).or_else(LF(error_read_file(fpath)));
};

// glob shader codes
void glob(SV const dir, SV const generated_extension, std::vector<Utils::BinDataOption>& output_codes) {
	for (auto const& entry : fs::directory_iterator(dir)) {
		auto const path_str = entry.path().string();
		auto const path_sv  = SV(path_str);
		if (entry.is_regular_file() && path_sv.ends_with(generated_extension)) {
			output_codes.push_back(readfile(path_sv));
		}
	}
};

template <typename... Args>
class Tu : std::tuple<Args...> {
	using Base = std::tuple<Args...>;
	// using Base::Base;
	using std::tuple<Args...>::std::tuple;
};

using CreatePipelineFN     = vk::Pipeline (BRDFSample::*)(vk::ShaderModule, const SpecData&);
using PipelineFromModuleFN = vk::Pipeline (*)(vk::ShaderModule, void* user_data);

struct _UserData {
	BRDFSample*      sample;
	CreatePipelineFN f;
	SpecData         spec;
};

[[nodiscard]]
auto PipelineFromCode(
	Utils::BinDataView             code,
	vk::Device                     device,
	vk::AllocationCallbacks const* allocator,
	PipelineFromModuleFN           create_pipeline_fn,
	void*                          user_data) -> vk::Pipeline {
	//
	vk::ShaderModuleCreateInfo info;
	vk::ShaderModule           module;
	info.codeSize = std::size(code);
	info.pCode    = reinterpret_cast<u32 const*>(std::data(code));
	CHECK_VULKAN_RESULT(device.createShaderModule(&info, allocator, &module));
	vk::Pipeline out_pipeline = create_pipeline_fn(module, user_data);
	device.destroyShaderModule(module, allocator);
	return out_pipeline;
}

auto const pipeline_from_module = PipelineFromModuleFN{
	+[](vk::ShaderModule module, void* user_data) static -> vk::Pipeline {
		_UserData* pdata = static_cast<_UserData*>(user_data);
		return ((pdata->sample)->*(pdata->f))(module, pdata->spec);
	}};

// auto BRDFSample::PipelineData::GetPipeline(u32 id) -> vk::Pipeline {

// };

auto BRDFSample::EnsurePipeline(u32 id) -> vk::Pipeline {
	auto&      gdata = generated_data[id];
	auto const pid   = gdata.pipeline_id;

	if (pid != PipelineData::kUndefined) {
		if (auto pipeline = generated_pipelines[pid]) {
			return pipeline;
		}
	}

	char path_buffer[1024];
	auto make_path = [&](std::string_view const fname) {
		auto const printed = std::snprintf(
			path_buffer, sizeof(path_buffer),
			GENERATED_DIR_RELATIVE "/%s.slang.spv",
			fname.data());

		return std::string_view(path_buffer, printed);
	};

	auto const s_code = gdata.code.value_or(readfile(make_path(GeneratedNames()[id].name)).value());
	// readfile

	_UserData udata{this, &BRDFSample::CreatePipeline, {.function_type = function_type, .function_id = id}};

	// std::printf("Creating pipeline\n");
	auto const new_pipeline = PipelineFromCode(s_code, device, GetAllocator(), pipeline_from_module, &udata);

	gdata.pipeline_id = [&] -> u32 {
		for (auto i : Utils::indices(std::size(generated_pipelines))) {
			auto& elem = generated_pipelines[i];
			if (elem == vk::Pipeline{}) {
				elem = new_pipeline;
				return i;
			}
		}
		assert(0 && "No space for pipeline");
		return -1;
	}();
	return new_pipeline;
}

auto BRDFSample::GeneratedPipeline(u32 id) -> vk::Pipeline {
	return EnsurePipeline(id);
}

void BRDFSample::CreatePipelines() {
	LOG_DEBUG("BRDFSample::CreatePipelines()");
	using Utils::make_string;

	auto const shader_codes_main = std::array{
		readfile("Shaders/BRDFMain-point.slang.spv"),
		readfile("Shaders/BRDFMain-env.slang.spv"),
	};
	static_assert(std::size(shader_codes_main) == kPipelineFallbackCount);

	auto const max_f_id = kMaxGeneratedPipelines - 1;
	if (function_id) {
		function_id = std::min(*function_id, static_cast<decltype(function_id)::value_type>(max_f_id));
	}

	auto gen_shader_modules =
		[&](
			std::span<Utils::BinDataOption const> shader_codes,
			std::span<vk::Pipeline>               out_pipelines,
			CreatePipelineFN                      create_pipeline_fn) {
			//

			auto const num_codes = std::min(std::size(shader_codes), std::size(out_pipelines));
			for (u32 i = 0; i < num_codes; ++i) {
				_UserData udata{this, create_pipeline_fn, {.function_type = function_type, .function_id = i}};
				out_pipelines[i] = PipelineFromCode(*shader_codes[i], device, GetAllocator(), pipeline_from_module, &udata);
			}
		};
	// gen_shader_modules(shader_codes_generated, pipelines_header, &BRDFSample::CreatePipeline);
	gen_shader_modules(shader_codes_main, pipelines_fallback, &BRDFSample::CreatePipeline);

	// Skybox
	if (hasattr(&BRDFSample::cubemap_folder_path)) {
		constexpr auto s_name = std::string_view{"Shaders/Skybox.slang.spv"};
		auto const     s_code = std::array{readfile(s_name)};
		auto const     s_out  = std::span{&skybox_pipeline, 1};
		gen_shader_modules(s_code, s_out, &BRDFSample::CreateSkyboxPipeline);
	}
}

auto BRDFSample::CreateSkyboxPipeline(vk::ShaderModule shader_module, SpecData const& info) -> vk::Pipeline {
	return CreateSkyboxPipelinePrivate(shader_module);
}
auto BRDFSample::CreateSkyboxPipelinePrivate(vk::ShaderModule shader_module) -> vk::Pipeline {

	vk::PipelineShaderStageCreateInfo const shader_stages[] = {
		{.stage = vk::ShaderStageFlagBits::eVertex, .module = shader_module, .pName = "vs_main"},
		{.stage = vk::ShaderStageFlagBits::eFragment, .module = shader_module, .pName = "ps_main"},
	};

	auto const vertex_input_state = vk::PipelineVertexInputStateCreateInfo{};

	auto const input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo{
		.flags                  = {},
		.topology               = vk::PrimitiveTopology::eTriangleList,
		.primitiveRestartEnable = vk::False,
	};

	auto const viewport_state = vk::PipelineViewportStateCreateInfo{
		.viewportCount = 1,
		.scissorCount  = 1,
	};

	auto const rasterization_state = vk::PipelineRasterizationStateCreateInfo{
		.cullMode  = vk::CullModeFlagBits::eNone,
		.frontFace = vk::FrontFace::eCounterClockwise,
		.lineWidth = 1.0f,
	};

	auto const multisample_state = vk::PipelineMultisampleStateCreateInfo{
		.rasterizationSamples = vk::SampleCountFlagBits::e1,
	};

	auto const depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo{
		.depthTestEnable  = vk::True,
		.depthWriteEnable = vk::False,
		.depthCompareOp   = vk::CompareOp::eLessOrEqual,
		.minDepthBounds   = 0.0f,
		.maxDepthBounds   = 1.0f,
	};

	auto const color_write_mask =
		vk::ColorComponentFlagBits::eR
		| vk::ColorComponentFlagBits::eG
		| vk::ColorComponentFlagBits::eB
		| vk::ColorComponentFlagBits::eA;

	auto const color_blend_attachment_state = vk::PipelineColorBlendAttachmentState{
		.blendEnable    = vk::False,
		.colorWriteMask = color_write_mask,
	};

	auto const color_blend_state = vk::PipelineColorBlendStateCreateInfo{
		.attachmentCount = 1,
		.pAttachments    = &color_blend_attachment_state,
	};

	vk::DynamicState const dynamic_states[] = {
		vk::DynamicState::eViewport,
		vk::DynamicState::eScissor,
	};

	auto const dynamic_state = vk::PipelineDynamicStateCreateInfo{
		.dynamicStateCount = static_cast<u32>(std::size(dynamic_states)),
		.pDynamicStates    = dynamic_states,
	};

	auto const pipeline_rendering_info = vk::PipelineRenderingCreateInfo{
		.viewMask                = 0,
		.colorAttachmentCount    = 1,
		.pColorAttachmentFormats = &swapchain.GetFormat(),
		.depthAttachmentFormat   = depth_image.GetFormat(),
	};

	auto const create_info = vk::GraphicsPipelineCreateInfo{
		.pNext               = &pipeline_rendering_info,
		.stageCount          = static_cast<u32>(std::size(shader_stages)),
		.pStages             = std::data(shader_stages),
		.pVertexInputState   = &vertex_input_state,
		.pInputAssemblyState = &input_assembly_state,
		.pViewportState      = &viewport_state,
		.pRasterizationState = &rasterization_state,
		.pMultisampleState   = &multisample_state,
		.pDepthStencilState  = &depth_stencil_state,
		.pColorBlendState    = &color_blend_state,
		.pDynamicState       = &dynamic_state,
		.layout              = pipeline_layout,
	};

	vk::Pipeline pipeline;
	CHECK_VULKAN_RESULT(device.createGraphicsPipelines(GetPipelineCache(), 1, &create_info, GetAllocator(), &pipeline));

	return pipeline;
}
auto BRDFSample::CreatePipeline(vk::ShaderModule shader_module, SpecData const& info) -> vk::Pipeline {
	LOG_DEBUG("BRDFSample::CreatePipeline()");

	// Specialization constant for type of inferencing function
	// BrdfFunctionType specialization_value = info.function_type;

	vk::SpecializationMapEntry const specialization_entries[] = {
		{
			.constantID = 0,
			.offset     = offsetof(SpecData, function_type),
			.size       = sizeof(info.function_type),
		},
		{
			.constantID = 1,
			.offset     = offsetof(SpecData, function_id),
			.size       = sizeof(info.function_id),
		},
	};
	auto const specialization_info = vk::SpecializationInfo{
		.mapEntryCount = std::size(specialization_entries),
		.pMapEntries   = specialization_entries,
		.dataSize      = sizeof(info),
		.pData         = &info,
	};

	// auto const shader_stages = std::array<vk::PipelineShaderStageCreateInfo, 2>{
	vk::PipelineShaderStageCreateInfo const shader_stages[] = {
		{.stage = vk::ShaderStageFlagBits::eVertex, .module = shader_module, .pName = "vs_main"},
		{.stage = vk::ShaderStageFlagBits::eFragment, .module = shader_module, .pName = "ps_main", .pSpecializationInfo = &specialization_info},
	};

	vk::VertexInputBindingDescription const vertex_input_binding_descriptions[] = {{
		.binding   = 0,
		.stride    = sizeof(Vertex),
		.inputRate = vk::VertexInputRate::eVertex,
	}};

	vk::VertexInputAttributeDescription const vertex_input_attribute_descriptions[] = {
		{.location = 0, .binding = 0, .format = vk::Format::eR32G32B32A32Sfloat, .offset = offsetof(Vertex, pos)},
		{.location = 1, .binding = 0, .format = vk::Format::eR32G32B32A32Sfloat, .offset = offsetof(Vertex, normal)},
	};

	auto const vertex_input_state = vk::PipelineVertexInputStateCreateInfo{
		.vertexBindingDescriptionCount   = std::size(vertex_input_binding_descriptions),
		.pVertexBindingDescriptions      = std::data(vertex_input_binding_descriptions),
		.vertexAttributeDescriptionCount = std::size(vertex_input_attribute_descriptions),
		.pVertexAttributeDescriptions    = std::data(vertex_input_attribute_descriptions),
	};

	auto const input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo{
		.flags                  = {},
		.topology               = vk::PrimitiveTopology::eTriangleList,
		.primitiveRestartEnable = vk::False,
	};

	auto const viewport_state = vk::PipelineViewportStateCreateInfo{
		.viewportCount = 1,
		.scissorCount  = 1,
	};

	auto const rasterization_state = vk::PipelineRasterizationStateCreateInfo{
		.cullMode  = vk::CullModeFlagBits::eBack,
		.frontFace = vk::FrontFace::eCounterClockwise,
		.lineWidth = 1.0f,
	};

	auto const multisample_state = vk::PipelineMultisampleStateCreateInfo{
		.rasterizationSamples = vk::SampleCountFlagBits::e1,
	};

	auto const depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo{
		.depthTestEnable  = vk::True,
		.depthWriteEnable = vk::True,
		.depthCompareOp   = vk::CompareOp::eLess,
		.minDepthBounds   = 0.0f,
		.maxDepthBounds   = 1.0f,
	};

	auto const color_write_mask =
		vk::ColorComponentFlagBits::eR
		| vk::ColorComponentFlagBits::eG
		| vk::ColorComponentFlagBits::eB
		| vk::ColorComponentFlagBits::eA;

	auto const color_blend_attachment_state = vk::PipelineColorBlendAttachmentState{
		.blendEnable    = vk::False,
		.colorWriteMask = color_write_mask,
	};

	auto const color_blend_state = vk::PipelineColorBlendStateCreateInfo{
		.attachmentCount = 1,
		.pAttachments    = &color_blend_attachment_state,
	};

	vk::DynamicState const dynamic_states[] = {
		vk::DynamicState::eViewport,
		vk::DynamicState::eScissor,
	};

	auto const dynamic_state = vk::PipelineDynamicStateCreateInfo{
		.dynamicStateCount = static_cast<u32>(std::size(dynamic_states)),
		.pDynamicStates    = dynamic_states,
	};

	auto const pipeline_rendering_info = vk::PipelineRenderingCreateInfo{
		.viewMask                = 0,
		.colorAttachmentCount    = 1,
		.pColorAttachmentFormats = &swapchain.GetFormat(),
		.depthAttachmentFormat   = depth_image.GetFormat(),
	};

	auto const create_info = vk::GraphicsPipelineCreateInfo{
		.pNext               = &pipeline_rendering_info,
		.stageCount          = static_cast<u32>(std::size(shader_stages)),
		.pStages             = std::data(shader_stages),
		.pVertexInputState   = &vertex_input_state,
		.pInputAssemblyState = &input_assembly_state,
		.pViewportState      = &viewport_state,
		.pRasterizationState = &rasterization_state,
		.pMultisampleState   = &multisample_state,
		.pDepthStencilState  = &depth_stencil_state,
		.pColorBlendState    = &color_blend_state,
		.pDynamicState       = &dynamic_state,
		.layout              = pipeline_layout,
	};

	vk::Pipeline pipeline;
	CHECK_VULKAN_RESULT(device.createGraphicsPipelines(GetPipelineCache(), 1, &create_info, GetAllocator(), &pipeline));

	return pipeline;
}
