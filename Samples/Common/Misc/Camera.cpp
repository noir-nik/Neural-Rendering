module;
module Camera;

import Math;

void Camera::setProj(float fov, int width, int height, float z_near, float z_far) {
	proj = width > height
			   ? perspectiveX(fov, (float)width / height, z_near, z_far)
			   : perspectiveY(fov, (float)height / width, z_near, z_far);

	this->fov    = fov;
	this->z_near = z_near;
	this->z_far  = z_far;
}

void Camera::updateProjectionViewInverse() {
	gpuCamera.projection_view_inv = proj * (view | affineInverse);
	// gpuCamera.projection_view_inv = proj * (view | inverse);
}
