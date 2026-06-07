
module;
#include "CheckResult.h"
#include "vulkan/vulkan_core.h"

module BRDFSample;

#if defined(WITH_UI) && WITH_UI
import imgui;
import imgui_impl_vulkan;
import imgui_impl_glfw;

#include "Log.h"

void ImGuiCheckVulkanResult(VkResult result) {
	CHECK_VULKAN_RESULT(result);
}

void BRDFSample::CreateImGui() {
	VkFormat colorFormats[] = {VK_FORMAT_R8G8B8A8_UNORM};
	VkFormat depthFormat    = VK_FORMAT_D32_SFLOAT;

	ImGui_ImplVulkan_InitInfo initInfo{
		.Instance       = instance,
		.PhysicalDevice = physical_device,
		.Device         = device,
		.QueueFamily    = queue_family_index,
		.Queue          = queue,
		.DescriptorPool = descriptor_pool,
		// .MinImageCount  = swapchain.surface_capabilities.minImageCount,
		// .ImageCount     = (u32)images.size(),
		.MinImageCount = 3,
		.ImageCount    = static_cast<uint32_t>(swapchain.GetImages().size()),
		// .MSAASamples         = (VkSampleCountFlagBits)std::min(device->physicalDevice->maxSamples, sampleCount),
		// .MSAASamples                 = VK_SAMPLE_COUNT_1_BIT,
		.PipelineCache    = pipeline_cache,
		.PipelineInfoMain = {
			.PipelineRenderingCreateInfo = {
				.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
				.colorAttachmentCount    = 1,
				.pColorAttachmentFormats = colorFormats,
				.depthAttachmentFormat   = depthFormat,
			},
		},
		.UseDynamicRendering = true,
		.Allocator           = reinterpret_cast<VkAllocationCallbacks const*>(GetAllocator()),
		.CheckVkResultFn     = ImGuiCheckVulkanResult,
	};
	ImGui_ImplGlfw_InitForVulkan(static_cast<GLFWwindow*>(window.GetHandle()), true);
	ImGui_ImplVulkan_Init(&initInfo);
}

void BRDFSample::ImGuiNewFrame() {
	ImGui_ImplVulkan_NewFrame();
	ImGui_ImplGlfw_NewFrame();
}

// void Command::DrawImGui(void* imDrawData) {
// 	ImGui_ImplVulkan_RenderDrawData(static_cast<ImDrawData*>(imDrawData), resource->buffer);
// }

void BRDFSample::ImGuiShutdown() {
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
}
#endif // WITH_UI