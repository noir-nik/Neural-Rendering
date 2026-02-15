module;
module Camera;

import Math;
import std;

Camera::Camera(CameraCreationInfo const& info) : focus(info.focus), fov(info.fov), z_near(info.z_near), z_far(info.z_far) {
	view = lookAt(info.position, focus, info.up) | affineInverse;
	// view = lookAt(info.position, focus, info.up);// | affineInverse;
	// view = lookAt(info.position, focus, info.up) ;//| inverse4x4;
}

void Camera::setProj(float fov, int width, int height, float z_near, float z_far) {
/* 	proj = width > height
			   ? perspectiveX(fov, (float)width / height, z_near, z_far)
			   : perspectiveY(fov, (float)height / width, z_near, z_far);
 */
	proj = perspective(fov, (float)width / height, z_near, z_far);
	// proj = inverse(proj);
	this->fov    = fov;
	this->z_near = z_near;
	this->z_far  = z_far;
}

void Camera::updateProjectionViewInverse() {
	gpuCamera.projection_view_inv = proj * (view | inverse);
	// gpuCamera.projection_view_inv =   (view) *inverse(proj) ;
	// gpuCamera.projection_view_inv =   (view) *(proj) ;
	// gpuCamera.projection_view_inv = (proj) * (view );
}

void Camera::moveWithCursor(float width, float height, float delta_x, float delta_y) {
	float3 camera_pos     = getPosition();
	float3 camera_forward = getForward();
	float3 camera_right   = getRight();
	float3 camera_up      = getUp();

	float3 camera_to_focus = focus - camera_pos;
	float  focus_distance  = dot(camera_to_focus, camera_forward);

	float aspect_ratio = width / height;
	float frustum_height, frustum_width;

	float half_tan_fov = std::tanf(math::DEG_TO_RAD * (fov * 0.5f));

	if (width > height) {
		frustum_width  = 2.0f * focus_distance * half_tan_fov;
		frustum_height = frustum_width / aspect_ratio;
	} else {
		frustum_height = 2.0f * focus_distance * half_tan_fov;
		frustum_width  = frustum_height * aspect_ratio;
	}

	float scale_x = frustum_width / width;
	float scale_y = -frustum_height / height;

	float3 movement = camera_right * (delta_x * scale_x) - camera_up * (delta_y * scale_y);

	view = view | translate(movement);
}
