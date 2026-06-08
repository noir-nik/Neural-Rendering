
module UI;
import std;
import imgui;
import imgui_impl_vulkan;
import imgui_impl_glfw;

namespace UI {

ImFontAtlas*  sharedFontAtlas = nullptr;
ImFont*       defaultFont     = nullptr;
ImGuiContext* setupContext    = nullptr;

auto const assets_dir = "../Common/Assets/Fonts";

static void SetStyle(ImGuiStyle& style) {
	style.WindowRounding                       = 5.000;
	style.WindowBorderSize                     = 0.000;
	style.ChildRounding                        = 4.000;
	style.ChildBorderSize                      = 0.000;
	style.PopupRounding                        = 4.000;
	style.PopupBorderSize                      = 0.000;
	style.FrameRounding                        = 4.000;
	style.IndentSpacing                        = 4.000;
	style.ScrollbarRounding                    = 7.000;
	style.GrabRounding                         = 6.000;
	style.TabRounding                          = 2.000;
	style.TabBarBorderSize                     = 0.000;
	style.TableAngledHeadersAngle              = 0.611;
	style.Colors[ImGuiCol_TextDisabled]        = ImVec4(0.502, 0.502, 0.502, 1.000);
	style.Colors[ImGuiCol_TextDisabled]        = ImVec4(0.502, 0.502, 0.502, 1.000);
	style.Colors[ImGuiCol_WindowBg]            = ImVec4(0.106, 0.106, 0.106, 1.000);
	style.Colors[ImGuiCol_ChildBg]             = ImVec4(0.137, 0.137, 0.137, 0.980);
	style.Colors[ImGuiCol_Border]              = ImVec4(0.094, 0.094, 0.094, 0.502);
	style.Colors[ImGuiCol_FrameBg]             = ImVec4(0.176, 0.176, 0.176, 1.000);
	style.Colors[ImGuiCol_FrameBgHovered]      = ImVec4(0.306, 0.306, 0.306, 1.000);
	style.Colors[ImGuiCol_FrameBgActive]       = ImVec4(0.133, 0.133, 0.133, 1.000);
	style.Colors[ImGuiCol_TitleBgActive]       = ImVec4(0.040, 0.040, 0.040, 1.000);
	style.Colors[ImGuiCol_MenuBarBg]           = ImVec4(0.212, 0.212, 0.212, 1.000);
	style.Colors[ImGuiCol_Button]              = ImVec4(0.122, 0.122, 0.120, 1.000);
	style.Colors[ImGuiCol_ButtonHovered]       = ImVec4(0.200, 0.200, 0.200, 1.000);
	style.Colors[ImGuiCol_ButtonActive]        = ImVec4(0.200, 0.200, 0.200, 1.000);
	style.Colors[ImGuiCol_Header]              = ImVec4(0.282, 0.282, 0.282, 1.000);
	style.Colors[ImGuiCol_HeaderHovered]       = ImVec4(0.227, 0.298, 0.392, 1.000);
	style.Colors[ImGuiCol_HeaderActive]        = ImVec4(0.227, 0.298, 0.392, 1.000);
	style.Colors[ImGuiCol_Separator]           = ImVec4(0.216, 0.216, 0.216, 0.502);
	style.Colors[ImGuiCol_SeparatorHovered]    = ImVec4(0.090, 0.090, 0.090, 0.502);
	style.Colors[ImGuiCol_SeparatorActive]     = ImVec4(0.090, 0.090, 0.090, 0.502);
	style.Colors[ImGuiCol_ResizeGrip]          = ImVec4(0.090, 0.090, 0.090, 0.502);
	style.Colors[ImGuiCol_ResizeGripHovered]   = ImVec4(0.090, 0.090, 0.090, 0.502);
	style.Colors[ImGuiCol_ResizeGripActive]    = ImVec4(0.090, 0.090, 0.090, 0.502);
	style.Colors[ImGuiCol_TabHovered]          = ImVec4(0.200, 0.200, 0.200, 1.000);
	style.Colors[ImGuiCol_Tab]                 = ImVec4(0.114, 0.114, 0.114, 1.000);
	style.Colors[ImGuiCol_TabSelected]         = ImVec4(0.200, 0.200, 0.200, 1.000);
	style.Colors[ImGuiCol_TabSelectedOverline] = ImVec4(0.486, 0.486, 0.486, 1.000);
	style.Colors[ImGuiCol_TabDimmedSelected]   = ImVec4(0.255, 0.255, 0.255, 1.000);
	// style.Colors[ImGuiCol_DockingPreview]      = ImVec4(0.710, 0.710, 0.710, 0.706);
	style.Colors[ImGuiCol_DragDropTarget] = ImVec4(0.973, 0.973, 0.973, 0.902);
}

void Init() {
	ImGui::CheckVersion();
	setupContext = ImGui::CreateContext();

	ImGuiIO& io     = ImGui::GetIO();
	sharedFontAtlas = io.Fonts;
	// io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	// io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
	// io.IniFilename = "assets/imgui.ini";
	// io.FontGlobalScale;
	auto& style = ImGui::GetStyle();

	// style.FontScaleMain = 1.5;
	SetStyle(style);

	// return;

	// io.Fonts->AddFontDefault();
	ImFontConfig config;
	config.MergeMode = true;

	float const fontSize = 16.0f;

	namespace fs = std::filesystem;

	if (std::filesystem::exists(assets_dir)) {
		for (auto const& entry : fs::directory_iterator(assets_dir)) {
			if (entry.path().filename() == "InterVariable.ttf") {
				defaultFont = sharedFontAtlas->AddFontFromFileTTF(entry.path().string().c_str(), fontSize);
				// sharedFontAtlas->AddFontDefault()
				if (defaultFont == nullptr) {
					// LOG_WARN("Failed to load font: {}", entry.path().string());
					defaultFont = sharedFontAtlas->AddFontDefault();
				}
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

	if (imguiStyle != nullptr) {
		ImGui::GetStyle() = *static_cast<ImGuiStyle*>(imguiStyle);
	}
}

void Context::SetCurrent() {
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
