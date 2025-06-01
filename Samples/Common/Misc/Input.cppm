export module Input;

import Window;
import Camera;
import Mouse;
import Math;

export void ProcessViewportInput(GLFWWindow const& window, Camera& camera, Mouse& mouse, float delta_x, float delta_y);
