
module;
#include "CheckResult.h"
#include "vulkan/vulkan_core.h"

module BRDFSample;

#if defined(WITH_UI) && WITH_UI
import imgui;
import imgui_impl_vulkan;
import imgui_impl_glfw;

#include "Log.h"

void BRDFSample::DrawUI() {

	int  selected_option = function_id.value_or(0); // or_else([]{return 0u;});
	auto option_counter  = int{0};
	using SV             = std::string_view;

	auto const list_option = [&](SV str) {
		const bool is_selected = (selected_option == option_counter);
		if (ImGui::Selectable(str.data(), is_selected)) {
			selected_option = option_counter;
		}
		if (is_selected) {
			ImGui::SetItemDefaultFocus();
		}
		++option_counter;
	};

	auto fmin = std::numeric_limits<float>::min();
	if (ImGui::BeginListBox("##", ImVec2(-fmin, -fmin))) {
#define BRDF_NAME(x) list_option(#x);
#include "BRDFModels.def"
#undef BRDF_NAME
		ImGui::EndListBox();
	}

	auto is_radio = bool{false};
	if (is_radio) {
		option_counter = {};
#define BRDF_NAME(x) ImGui::RadioButton(#x, &selected_option, option_counter++);
#include "BRDFModels.def"
#undef BRDF_NAME
	}
	function_id = selected_option;

	// ImGui::Begin("Numbered Window", nullptr, ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoTitleBar);
	// // Calculate top-right position with padding
	float padding = 10.0f;
	float width   = 200.0f; // Define your window width
	float height  = 100.0f; // Define your window height

	// float pos_x = ImGui::GetMainViewport()->WorkSize.x - width - padding;
	float pos_x = padding;
	float pos_y = padding;

	ImGui::SetNextWindowPos(ImVec2(pos_x, pos_y), ImGuiCond_Always);
	ImGui::SetNextWindowSize(ImVec2(width, height), ImGuiCond_Always);

	static auto last_gpu_fps = u32(1000.0f / elapsed_last_frame_ms);

	static auto last_time = std::chrono::steady_clock::now();
	auto const  now       = std::chrono::steady_clock::now();
	auto const  elapsed   = now - last_time;

	auto const update_interval = 200u;

	if (elapsed > std::chrono::duration<float, std::milli>(update_interval)) {
		last_gpu_fps = u32(1000.0f / elapsed_last_frame_ms);
		last_time    = now;
	}

	auto invisible_flags{
		ImGuiWindowFlags_NoBackground
		| ImGuiWindowFlags_NoTitleBar
		| ImGuiWindowFlags_NoCollapse
		| ImGuiWindowFlags_NoResize};

	ImGui::Begin("Numbered Window", nullptr, invisible_flags);
	ImGui::PushFont(nullptr, 32);
	ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "%u FPS", last_gpu_fps);
	ImGui::PopFont();
	ImGui::End();
}
#endif // WITH_UI
