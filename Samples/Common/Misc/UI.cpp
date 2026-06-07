
module UI;
import std;
import imgui;
import imgui_impl_vulkan;
import imgui_impl_glfw;
 


namespace UI {

ImFontAtlas* sharedFontAtlas = nullptr;
ImFont* defaultFont = nullptr;
ImGuiContext* setupContext = nullptr;

void Init(){
	ImGui::CheckVersion();
	setupContext = ImGui::CreateContext();
	return;
	
	ImGuiIO& io = ImGui::GetIO();
	sharedFontAtlas = io.Fonts;
	// io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	// io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
	io.IniFilename = "assets/imgui.ini";

	// io.Fonts->AddFontDefault();
	ImFontConfig config;
	config.MergeMode = true;

	float const fontSize = 15.0f;
	for (auto const& entry : std::filesystem::directory_iterator("assets/fonts")) {
		if (entry.path().extension() == ".ttf" || entry.path().extension() == ".otf") {
			// auto font = io.Fonts->AddFontFromFileTTF(entry.path().string().c_str(), fontSize);
			// if (font && entry.path().filename() == "InterVariable.ttf") {
			// 	io.FontDefault = font;
			// }
			if (entry.path().filename() == "InterVariable.ttf") {
				defaultFont = sharedFontAtlas->AddFontFromFileTTF(entry.path().string().c_str(), fontSize);
				if (defaultFont == nullptr) {
					// LOG_WARN("Failed to load font: {}", entry.path().string());
					defaultFont = sharedFontAtlas->AddFontDefault();
				}
			} else {
				config.GlyphMinAdvanceX = 13.0f; // Use if you want to make the icon monospaced
				// static const ImWchar icon_ranges[] = { ICON_MIN_FA, ICON_MAX_FA, 0 };
				io.Fonts->AddFontFromFileTTF(entry.path().string().c_str(), 13.0f/* , &config */);
			}
		}
	}
	io.FontDefault = defaultFont;

}

void Destroy() {
	ImGui::SetCurrentContext(setupContext);
	// vb::ImGuiShutdown();
	ImGui::DestroyContext(setupContext);
}


void Context::Init(void* imguiStyle) {
	context = ImGui::CreateContext(sharedFontAtlas);
	SetCurrent();
	auto& io = ImGui::GetIO();
	// io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	// io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
	io.FontDefault = defaultFont;
	io.IniFilename = "assets/imgui.ini";

	if (imguiStyle != nullptr){
		ImGui::GetStyle() = *static_cast<ImGuiStyle*>(imguiStyle);
	}
}

void Context::SetCurrent(){
	ImGui::SetCurrentContext(static_cast<ImGuiContext*>(context));
}

void ImGuiShutdown() {
	ImGui_ImplVulkan_Shutdown();
	ImGui_ImplGlfw_Shutdown();
}

void Context::Destroy() {
	SetCurrent();
	ImGuiShutdown();
	ImGui::DestroyContext(static_cast<ImGuiContext*>(context));
}



} // namespace UI