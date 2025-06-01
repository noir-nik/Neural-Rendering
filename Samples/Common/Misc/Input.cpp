module Input;

import Window;
import Camera;
import Mouse;
import Math;
import Glfw;
import std;

using namespace math;
using u32 = std::uint32_t;

void ProcessViewportInput(GLFWWindow const& window, Camera& camera, Mouse& mouse, float delta_x, float delta_y) {
	using namespace Glfw;

	auto&       camera_pos     = camera.getPosition();
	auto const& camera_right   = camera.getRight();
	auto const& camera_up      = camera.getUp();
	auto const& camera_forward = camera.getForward();

	vec2 delta_pos = {mouse.delta_x, mouse.delta_y};
	// vec2 delta_pos = {mouse.delta_x, mouse.delta_y};

	GLFWwindow* glfw_window = static_cast<GLFWwindow*>(window.GetHandle());

	u32 const pressed_buttons_count =
		std::ranges::fold_left(
			std::span(mouse.button_state) | std::views::take(3 /* left, middle, right */),
			0, [](u32 state, Action action) {
				return state + (action == Action::ePress);
			});

	if (pressed_buttons_count > 1) [[unlikely]]
		return;
	auto button_pressed = [&](MouseButton button) { return mouse.button_state[std::to_underlying(button)] == Action::ePress; };

	if (button_pressed(MouseButton::eRight)) {
		// if (glfwGetKey(glfw_window, std::to_underlying(Glfw::Key::eLeftAlt)) == std::to_underlying(Glfw::Action::ePress)) {
		if (mouse.is_dragging) {
			auto zoom_factor = camera.zoom_factor * length(camera_pos - camera.focus);
			auto movement    = (zoom_factor * delta_pos.x) * camera_forward;
			camera.view      = camera.view | translate(movement);
			// camera.focus += movement;
		} else {
			// all float3
			camera_pos -= camera.focus;

			// Correct upside down
			// float3 world_up = float3(0.0f, 0.0f, 1.0f);
			// float rotation_sign = dot(camera_up, world_up) < 0.0f ? -1.0f : 1.0f;
			float rotation_sign = 1.0f;

			camera.view = camera.view
						  | rotate(camera_right, -delta_pos.y * camera.rotation_factor)
						  | rotate(camera_up, rotation_sign * delta_pos.x * camera.rotation_factor);
			camera_pos += camera.focus;
		}
		// window->AddFramesToDraw(1);
		// camera.updateProjView();
		return;
	}

	if (button_pressed(MouseButton::eMiddle)) {

		int x, y, width, height;
		window.GetRect(x, y, width, height);
		camera.moveWithCursor(width, height, delta_pos.x, delta_pos.y);
	}

	if (button_pressed(MouseButton::eLeft)) {
		// window->AddFramesToDraw(1);
		return;
	}
}
