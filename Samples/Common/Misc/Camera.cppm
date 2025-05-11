module;

export module Camera;

import Math;

using namespace math;

export extern "C++" {
#include "ShaderTypes.h"
}

export struct CameraCreationInfo {
	float3 position = {1.0f, 3.0f, 5.0f};
	float3 focus    = {0.0f, 0.0f, 0.0f};
	float3 up       = {0.0f, 1.0f, 0.0f};

	float fov    = 50.0f;
	float z_near = 0.01f;
	float z_far  = 1000.0f;
};

export struct Camera {
	static constexpr inline float rotation_factor = 0.0025f;
	static constexpr inline float zoom_factor     = 0.01f;
	static constexpr inline float move_factor     = 0.00107f;

	mat4      view;
	mat4      proj;
	GPUCamera gpuCamera;

	float3 focus  = {0.0f, 0.0f, 0.0f};
	float  fov    = 50.0f;
	float  z_near = 0.01f;
	float  z_far  = 1000.0f;

	Camera(CameraCreationInfo const& info);

	inline auto getRight() -> float3& { return view.col(0).xyz; }
	inline auto getUp() -> float3& { return view.col(1).xyz; }
	inline auto getForward() -> float3& { return view.col(2).xyz; }
	inline auto getPosition() -> float3& { return view.col(3).xyz; }

	inline auto getRight() const -> float3 const& { return view.col(0).xyz; }
	inline auto getUp() const -> float3 const& { return view.col(1).xyz; }
	inline auto getForward() const -> float3 const& { return view.col(2).xyz; }
	inline auto getPosition() const -> float3 const& { return view.col(3).xyz; }

	inline auto getProj() const -> mat4 const& { return proj; }
	inline auto getView() const -> mat4 const& { return view; }
	inline auto getProjViewInv() const -> mat4 const& { return gpuCamera.projection_view_inv; }
	inline auto getFocus() const -> float3 const& { return focus; }

	void setProj(float fov, int width, int height, float z_near, float z_far);

	void moveWithCursor(float width, float height, float delta_x, float delta_y);

	inline void updateProjection(int width, int height) {
		setProj(fov, width, height, z_near, z_far);
	}

	void updateProjectionViewInverse();
};
