module;

export module Camera;

import Math;

using namespace math;

export extern "C++" {
#include "ShaderTypes.h"
}

export struct Camera {
	float static constexpr inline rotation_factor = 0.0025f;
	float static constexpr inline zoom_factor     = 0.01f;
	float static constexpr inline move_factor     = 0.00107f;

	mat4      view;
	mat4      proj;
	GPUCamera gpuCamera;

	vec3  focus;
	float fov   = 50.0f;
	float z_near = 0.01f;
	float z_far  = 1000.0f;

	inline Camera(vec3 const& position = vec3(0.5f, 3.0f, 5.0f),
				  vec3 const& focus    = vec3(0.0f, 0.0f, 0.0f),
				  vec3 const& up       = vec3(0.0f, 1.0f, 0.0f))
		: focus(focus) {
		view = lookAt(position, focus, up) | affineInverse;
	}

	inline auto getRight() -> vec3& { return view.col(0).xyz; }
	inline auto getUp() -> vec3& { return view.col(1).xyz; }
	inline auto getForward() -> vec3& { return view.col(2).xyz; }
	inline auto getPosition() -> vec3& { return view.col(3).xyz; }

	inline auto getRight() const -> vec3 const& { return view.col(0).xyz; }
	inline auto getUp() const -> vec3 const& { return view.col(1).xyz; }
	inline auto getForward() const -> vec3 const& { return view.col(2).xyz; }
	inline auto getPosition() const -> vec3 const& { return view.col(3).xyz; }

	inline auto getProj() const -> mat4 const& { return proj; }
	inline auto getView() const -> mat4 const& { return view; }
	inline auto getProjViewInv() const -> mat4 const& { return gpuCamera.projection_view_inv; }
	inline auto getFocus() const -> vec3 const& { return focus; }

	void setProj(float fov, int width, int height, float z_near, float z_far);

	inline void updateProjection(int width, int height) {
		setProj(fov, width, height, z_near, z_far);
	}

	void updateProjectionViewInverse();
};
