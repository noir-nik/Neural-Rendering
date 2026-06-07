
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

	if (ImGui::BeginListBox("##")) {
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
}
#endif // WITH_UI
