/**
 * @file HdriToCubemap.hpp
 *
 * @brief Simple single-header library to convert an equirectangular hdri image to cubemap images.
 *
 * @author Ingvar Out
 * Contact: ivarout2@gmail.com
 *
 * @date November 22, 2020
 */

/*
 MIT License

Copyright (c) [2020] [Ingvar Out]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

module;
#include <stdio.h> // stderr
export module HdriToCubemap;

import std;
import StbModule;

// static constexpr auto M_PI = 3.14159265358979323846;
static constexpr auto M_PI = std::numbers::pi;

#define LOG_WARN(fmt, ...) \
	std::fprintf(stderr, fmt, __VA_ARGS__)

#define LIFT(fn) [&](auto&&... args) { return fn(args...); }

#define ADD_PROPERTY(name, type) \
public: \
	inline auto get_##name() const -> type { return name##_; } \
\
	inline auto get_##name##_ref() const -> type const& { return name##_; } \
	inline auto get_##name##_ptr() const -> type const* { return &name##_; } \
\
protected: \
	inline auto get_##name##_ref() -> type& { return name##_; } \
	inline auto get_##name##_ptr() -> type* { return &name##_; } \
	inline auto set_##name(const type name) -> decltype(*this)& { \
		this->name##_ = name; \
		return *this; \
	} \
\
private: \
	type name##_

class HdriToCubemapBase {

	ADD_PROPERTY(is_hdri, bool);
	ADD_PROPERTY(width, int);
	ADD_PROPERTY(height, int);
	ADD_PROPERTY(num_channels, int);
	ADD_PROPERTY(filter_linear, bool);
	ADD_PROPERTY(resolution, int);

public:
	static constexpr std::size_t kNumFaces = 6;

	auto image_size() const -> std::size_t { return get_resolution() * get_resolution() * get_num_channels(); }
	auto size() const -> std::size_t { return image_size() * kNumFaces; }

	constexpr inline auto get_num_faces() const -> std::size_t { return kNumFaces; }

	void write_cubemap(std::string_view const output_folder, std::span<void const* const> faces);
};

export template <typename T>
class HdriToCubemap : public HdriToCubemapBase {
public:
	using size_type = T;

	auto init(std::string_view file_location, int desired_resolution, bool filter_linear = true) -> bool;
	void destroy();

	auto get_faces() -> std::span<T*> { return {std::data(faces_), std::size(faces_)}; }
	T*   get_front() { return faces_[0]; }
	T*   get_back() { return faces_[1]; }
	T*   get_left() { return faces_[2]; }
	T*   get_right() { return faces_[3]; }
	T*   get_up() { return faces_[4]; }
	T*   get_down() { return faces_[5]; }
	void write_cubemap(std::string_view const output_folder = "");

	auto image_size_bytes() const -> std::size_t { return image_size() * sizeof(size_type); }
	auto size_bytes() const -> std::size_t { return image_size_bytes() * get_num_faces(); }

	auto is_valid() -> bool;

private:
	void calculate_cubemap();

private:
	T* image_data_;

	std::array<T*, kNumFaces> faces_;

	static constexpr auto desired_channels_ = int{4};
};

template <>
auto HdriToCubemap<unsigned char>::init(std::string_view pathHdri, int desired_resolution, bool filter_linear) -> bool {

	set_resolution(desired_resolution);
	set_filter_linear(filter_linear);

	stbi_set_flip_vertically_on_load(1);
	set_is_hdri(stbi_is_hdr(pathHdri.data()));

	if (get_is_hdri())
		LOG_WARN("%s", "Warning: image will be converted from hdr to ldr by stb_image. Use float-type template argument to create an hdr cubemap\n");
	set_is_hdri(false);

	image_data_            = stbi_load(pathHdri.data(), get_width_ptr(), get_height_ptr(), get_num_channels_ptr(), desired_channels_);
	get_num_channels_ref() = desired_channels_;
	if (!image_data_) {
		LOG_WARN("Failed to load image %*s\n", static_cast<int>(pathHdri.size()), pathHdri.data());
		return false;
	}

	for (int i = 0; i < 6; i++)
		faces_[i] = new unsigned char[image_size()];

	calculate_cubemap();
	return true;
}
template <>
auto HdriToCubemap<float>::init(std::string_view pathHdri, int desired_resolution, bool filter_linear) -> bool {
	set_resolution(desired_resolution);
	set_filter_linear(filter_linear);

	set_is_hdri(stbi_is_hdr(pathHdri.data()));

	if (!get_is_hdri())
		LOG_WARN("%s", "Warning: image will be converted from ldr to hdr by stb_image. Use unsigned-char-type template argument to create an ldr cubemap\n");
	set_is_hdri(true);

	image_data_            = stbi_loadf(pathHdri.data(), get_width_ptr(), get_height_ptr(), get_num_channels_ptr(), desired_channels_);
	get_num_channels_ref() = desired_channels_;

	if (!image_data_) {
		LOG_WARN("Failed to load image %*s\n", static_cast<int>(pathHdri.size()), pathHdri.data());
		return false;
	}

	for (int i = 0; i < 6; i++)
		faces_[i] = new float[image_size()];

	calculate_cubemap();
	return true;
}

template <typename T>
void HdriToCubemap<T>::destroy() {
	if (is_valid()) {
		stbi_image_free(image_data_);
		for (auto f : get_faces()) {
			delete[] f;
		}
	}
}

template <typename T>
void HdriToCubemap<T>::write_cubemap(std::string_view const output_folder) {
	auto pdata = reinterpret_cast<void const* const*>(std::data(faces_));
	auto fspan = std::span{pdata, std::size(faces_)};
	HdriToCubemapBase::write_cubemap(output_folder, fspan);
}

template <typename T>
auto HdriToCubemap<T>::is_valid() -> bool {
	return std::ranges::all_of(get_faces(), LIFT(bool));
};

template <typename T>
void HdriToCubemap<T>::calculate_cubemap() {
	struct Vec3 {
		float x, y, z;
	};
	std::array<std::array<Vec3, 3>, 6> const startRightUp = {{
		// for each face, contains the 3d starting point (corresponding to left bottom pixel), right direction, and up direction in 3d space, correponding to pixel x,y coordinates of each face		{{-1.0f, -1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},
		{{{-1.0f, -1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}}, // front
		{{{1.0f, -1.0f, 1.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}},  // back
		{{{-1.0f, -1.0f, 1.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f, 0.0f}}}, // left
		{{{1.0f, -1.0f, -1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}}},  // right
		{{{-1.0f, 1.0f, -1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}}},  // up
		{{{-1.0f, -1.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f}}}  // down
	}};

	for (int i = 0; i < 6; i++) {
		Vec3 const& start = startRightUp[i][0];
		Vec3 const& right = startRightUp[i][1];
		Vec3 const& up    = startRightUp[i][2];

		T*   face = faces_[i];
		Vec3 pixelDirection3d; // 3d direction corresponding to a pixel in the cubemap face
		// #pragma omp parallel for (private pixelDirection?)
		for (int row = 0; row < get_resolution(); row++) {
			for (int col = 0; col < get_resolution(); col++) {
				pixelDirection3d.x = start.x + ((float)col * 2.0f + 0.5f) / (float)get_resolution() * right.x + ((float)row * 2.0f + 0.5f) / (float)get_resolution() * up.x;
				pixelDirection3d.y = start.y + ((float)col * 2.0f + 0.5f) / (float)get_resolution() * right.y + ((float)row * 2.0f + 0.5f) / (float)get_resolution() * up.y;
				pixelDirection3d.z = start.z + ((float)col * 2.0f + 0.5f) / (float)get_resolution() * right.z + ((float)row * 2.0f + 0.5f) / (float)get_resolution() * up.z;

				float azimuth   = std::atan2f(pixelDirection3d.x, -pixelDirection3d.z) + (float)M_PI; // add pi to move range to 0-360 deg
				float elevation = std::atanf(pixelDirection3d.y / std::sqrtf(pixelDirection3d.x * pixelDirection3d.x + pixelDirection3d.z * pixelDirection3d.z)) + (float)M_PI / 2.0f;

				float colHdri = (azimuth / (float)M_PI / 2.0f) * get_width(); // add pi to azimuth to move range to 0-360 deg
				float rowHdri = (elevation / (float)M_PI) * get_height();

				if (!get_filter_linear()) {
					int colNearest = std::clamp((int)colHdri, 0, get_width() - 1);
					int rowNearest = std::clamp((int)rowHdri, 0, get_height() - 1);

					face[col * get_num_channels() + get_resolution() * row * get_num_channels()]     = image_data_[colNearest * get_num_channels() + get_width() * rowNearest * get_num_channels()];     // red
					face[col * get_num_channels() + get_resolution() * row * get_num_channels() + 1] = image_data_[colNearest * get_num_channels() + get_width() * rowNearest * get_num_channels() + 1]; // green
					face[col * get_num_channels() + get_resolution() * row * get_num_channels() + 2] = image_data_[colNearest * get_num_channels() + get_width() * rowNearest * get_num_channels() + 2]; // blue
					if (get_num_channels() > 3)
						face[col * get_num_channels() + get_resolution() * row * get_num_channels() + 3] = image_data_[colNearest * get_num_channels() + get_width() * rowNearest * get_num_channels() + 3]; // alpha
				} else                                                                                                                                                                                       // perform bilinear interpolation
				{
					float intCol, intRow;
					float factorCol = std::modf(colHdri - 0.5f, &intCol); // factor gives the contribution of the next column, while the contribution of intCol is 1 - factor
					float factorRow = std::modf(rowHdri - 0.5f, &intRow);

					int low_idx_row    = static_cast<int>(intRow);
					int low_idx_column = static_cast<int>(intCol);
					int high_idx_column;
					if (factorCol < 0.0f) // modf can only give a negative value if the azimuth falls in the first pixel, left of the center, so we have to mix with the pixel on the opposite side of the panoramic image
						high_idx_column = get_width() - 1;
					else if (low_idx_column == get_width() - 1) // if we are in the right-most pixel, and fall right of the center, mix with the left-most pixel
						high_idx_column = 0;
					else
						high_idx_column = low_idx_column + 1;

					int high_idx_row;
					if (factorRow < 0.0f)
						high_idx_row = get_height() - 1;
					else if (low_idx_row == get_height() - 1)
						high_idx_row = 0;
					else
						high_idx_row = low_idx_row + 1;

					factorCol = std::abs(factorCol);
					factorRow = std::abs(factorRow);
					float f1  = (1 - factorRow) * (1 - factorCol);
					float f2  = factorRow * (1 - factorCol);
					float f3  = (1 - factorRow) * factorCol;
					float f4  = factorRow * factorCol;

					for (int j = 0; j < get_num_channels(); j++) {
						unsigned char interpolatedValue = static_cast<unsigned char>(image_data_[low_idx_column * get_num_channels() + get_width() * low_idx_row * get_num_channels() + j] * f1 + image_data_[low_idx_column * get_num_channels() + get_width() * high_idx_row * get_num_channels() + j] * f2 + image_data_[high_idx_column * get_num_channels() + get_width() * low_idx_row * get_num_channels() + j] * f3 + image_data_[high_idx_column * get_num_channels() + get_width() * high_idx_row * get_num_channels() + j] * f4);

						face[col * get_num_channels() + get_resolution() * row * get_num_channels() + j] = std::clamp(interpolatedValue, (std::uint8_t)0, (std::uint8_t)255);
					}
				}
			}
		}
	}
}
