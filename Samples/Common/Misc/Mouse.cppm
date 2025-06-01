module;

export module Mouse;
import Glfw;
import std;

export struct Mouse {
	float x = 300.0f;
	float y = 300.0f;

	float        delta_x                                                                               = 0.0f;
	float        delta_y                                                                               = 0.0f;
	Glfw::Action button_state[std::underlying_type_t<Glfw::MouseButton>(Glfw::MouseButton::eLast) + 1] = {};

	bool is_dragging = false;
	void StartDragging() { is_dragging = true; }
	void StopDragging() { is_dragging = false; }
};
