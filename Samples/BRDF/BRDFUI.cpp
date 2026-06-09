
module;
#include "CheckResult.h"
#include "vulkan/vulkan_core.h"

module BRDFSample;

#if defined(WITH_UI) && WITH_UI
import imgui;
import imgui_impl_vulkan;
import imgui_impl_glfw;

#include "Log.h"
using SV                        = std::string_view;
static constexpr auto kFloatMin = std::numeric_limits<float>::min();

static constexpr auto invisible_flags{
	ImGuiWindowFlags_NoBackground
	| ImGuiWindowFlags_NoTitleBar
	| ImGuiWindowFlags_NoCollapse
	| ImGuiWindowFlags_NoResize
	| ImGuiWindowFlags_NoScrollbar
	| ImGuiWindowFlags_NoScrollWithMouse
	//
};

auto color_red   = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
auto color_green = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);

// static constexpr auto gImColors = std::array

#ifdef TW_COLOR
#undef TW_COLOR
#endif

namespace TWImColor {
enum class TWImColor {
#define TW_COLOR(name, hex, r, g, b) name,
#include "TWColors.def"
#undef TW_COLOR
};

constexpr auto GetTWImColor(TWImColor color) {
	switch (color) {
#define TW_COLOR(name, hex, r, g, b) \
	case TWImColor::name: return ImVec4(r / 256.0f, g / 256.0f, b / 256.0f, 1.0);
#include "TWColors.def"
#undef TW_COLOR
	}
};

#define TW_COLOR(name, hex, r, g, b) \
	static constexpr auto name = ImVec4(r / 256.0f, g / 256.0f, b / 256.0f, 1.0);
#include "TWColors.def"
#undef TW_COLOR

} // namespace TWImColor

// COLOR(Gray100, #f7fafc, 247, 250, 252)

auto ModelsWindow(u32 old_selected_option) -> u32 {
	float const padding = 10.0f;
	float const width   = 400.0f; // Define your window width
	float const height  = 300.0f; // Define your window height
	float const pos_x   = ImGui::GetMainViewport()->WorkSize.x - width - padding;
	float const pos_y   = padding;

	int  selected_option = old_selected_option; // or_else([]{return 0u;});
	auto option_counter  = int{0};

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

	ImGui::SetNextWindowPos(ImVec2(pos_x, pos_y), ImGuiCond_Once);
	ImGui::SetNextWindowSize(ImVec2(width, height), ImGuiCond_Once);

	if (ImGui::Begin("Models")) {
		if (ImGui::BeginListBox("##", ImVec2(-kFloatMin, -kFloatMin))) {
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
		// function_id = selected_option;
		ImGui::End();
	}
	if (selected_option != old_selected_option) {
		std::printf("Model %u\n", selected_option);
	}
	return selected_option;
}

auto FPSWindow(float elapsed_last_frame_ms) -> void {

	float padding = 10.0f;
	float width   = 400.0f;
	float height  = 200.0f;

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

	if (ImGui::Begin("FPS", nullptr, invisible_flags)) {
		ImGui::PushFont(nullptr, 96);
		ImGui::TextColored(color_green, "%u FPS", last_gpu_fps);
		ImGui::PopFont();
		ImGui::End();
	}
}
auto DetailsWindow(u32 selected_option, CSpan<BRDFModelData> models) -> void {

	float padding = 10.0f;
	float width   = 400.0f;
	float height  = 200.0f;

	using namespace TWImColor;

	float pos_x = ImGui::GetMainViewport()->WorkSize.x - width - padding;
	// float pos_x = padding;
	float pos_y = padding;

	ImGui::SetNextWindowPos(ImVec2(pos_x, pos_y), ImGuiCond_Always);
	ImGui::SetNextWindowSize(ImVec2(width, height), ImGuiCond_Always);

	if (ImGui::Begin("Details", nullptr, invisible_flags)) {
		ImGui::PushFont(nullptr, 64);
		ImGui::TextColored(color_green, "%s", std::data(models[selected_option].type)); // Type: 
		ImGui::TextColored(color_green, "%u", (models[selected_option].learnable_params)); // Parameters: 
		ImGui::PopFont();
		ImGui::End();
	}
}
void BRDFSample::DrawUI() {
	auto models = GeneratedNames();

	if (is_models_visible) {
		function_id = ModelsWindow(function_id.value_or(0));
	}
	if (is_fps_visible) {
		FPSWindow(elapsed_last_frame_ms);
	}
	DetailsWindow(*function_id, models);
}
#endif // WITH_UI
