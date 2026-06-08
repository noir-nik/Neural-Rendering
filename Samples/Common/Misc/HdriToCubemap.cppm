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
// #include <cmath>
export module HdriToCubemap;

import std;
import StbModule;

// static constexpr auto M_PI = 3.14159265358979323846;
static constexpr auto M_PI = std::numbers::pi;

export template <typename T>
class HdriToCubemap {
public:
	auto Init(std::string_view fileLocation, int cubemapResolution, bool filterLinear = true) -> bool;
	void Destroy();

	bool isHdri() const { return m_isHdri; }
	int  getCubemapResolution() const { return m_cubemapResolution; }
	int  getNumChannels() const { return m_channels; }
	T**  getFaces() { return m_faces; }
	T*   getFront() { return m_faces[0]; }
	T*   getBack() { return m_faces[1]; }
	T*   getLeft() { return m_faces[2]; }
	T*   getRight() { return m_faces[3]; }
	T*   getUp() { return m_faces[4]; }
	T*   getDown() { return m_faces[5]; }
	void writeCubemap(const std::string& outputFolder = "");

private:
	void calculateCubemap();

private:
	bool m_isHdri;
	int  m_width, m_height, m_channels;
	bool m_filterLinear;
	T*   m_imageData;
	T**  m_faces;
	int  m_cubemapResolution;
};

/* template <typename T>
auto HdriToCubemap<T>::Initsf(std::string_view pathHdri, int cubemapResolution, bool filterLinear) -> bool {
	m_cubemapResolution = cubemapResolution;
	m_filterLinear      = filterLinear;

	stbi_set_flip_vertically_on_load(1);
	m_isHdri = stbi_is_hdr(pathHdri.data());
	if (m_isHdri)
		std::printf("%s", "Warning: image will be converted from hdr to ldr by stb_image. Use float-type template argument to create an hdr cubemap\n");
	m_isHdri = false;
}
 */
template <>
auto HdriToCubemap<unsigned char>::Init(std::string_view pathHdri, int cubemapResolution, bool filterLinear) -> bool {

	m_cubemapResolution = cubemapResolution;
	m_filterLinear      = filterLinear;

	stbi_set_flip_vertically_on_load(1);
	m_isHdri = stbi_is_hdr(pathHdri.data());
	if (m_isHdri)
		std::printf("%s", "Warning: image will be converted from hdr to ldr by stb_image. Use float-type template argument to create an hdr cubemap\n");
	m_isHdri = false;

	m_imageData = stbi_load(pathHdri.data(), &m_width, &m_height, &m_channels, 0);
	if (!m_imageData) {
		std::printf("Failed to load image %*s\n", static_cast<int>(pathHdri.size()), pathHdri.data());
		return false;
	}

	m_faces = new unsigned char*[6];
	for (int i = 0; i < 6; i++)
		m_faces[i] = new unsigned char[m_cubemapResolution * m_cubemapResolution * m_channels];

	calculateCubemap();
	return true;
}
template <>
auto HdriToCubemap<float>::Init(std::string_view pathHdri, int cubemapResolution, bool filterLinear) -> bool {
	m_cubemapResolution = cubemapResolution;
	m_filterLinear      = filterLinear;

	m_isHdri = stbi_is_hdr(pathHdri.data());
	if (!m_isHdri)
		std::printf("%s", "Warning: image will be converted from ldr to hdr by stb_image. Use unsigned-char-type template argument to create an ldr cubemap\n");
	m_isHdri = true;

	m_imageData = stbi_loadf(pathHdri.data(), &m_width, &m_height, &m_channels, 0);
	if (!m_imageData) {
		std::printf("Failed to load image %*s\n", static_cast<int>(pathHdri.size()), pathHdri.data());
		return false;
	}

	m_faces = new float*[6];
	for (int i = 0; i < 6; i++)
		m_faces[i] = new float[m_cubemapResolution * m_cubemapResolution * m_channels];

	calculateCubemap();
	return true;
}

template <typename T>
void HdriToCubemap<T>::Destroy() {
	stbi_image_free(m_imageData);
	for (int i = 0; i < 6; i++)
		delete[] *m_faces++;
}

template <typename T>
void HdriToCubemap<T>::writeCubemap(const std::string& outputFolder) {
	stbi_flip_vertically_on_write(1);
	std::vector<std::string> filenames = {"front", "back", "left", "right", "up", "down"};
	for (int i = 0; i < 6; i++) {
		int success;

		std::string path = outputFolder + (outputFolder.empty() ? "" : "/") + filenames[i];
		if (m_isHdri) {
			path += ".hdr";
			success = stbi_write_hdr(path.c_str(), m_cubemapResolution, m_cubemapResolution, m_channels, (const float*)m_faces[i]);
		} else {
			path += ".png";
			success = stbi_write_png(path.c_str(), m_cubemapResolution, m_cubemapResolution, m_channels, (const unsigned char*)m_faces[i], 0);
		}
		if (!success)
			std::cout << "Warning: could not write '" << path << "'";
	}
}

template <typename T>
void HdriToCubemap<T>::calculateCubemap() {
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

		T*   face = m_faces[i];
		Vec3 pixelDirection3d; // 3d direction corresponding to a pixel in the cubemap face
		// #pragma omp parallel for (private pixelDirection?)
		for (int row = 0; row < m_cubemapResolution; row++) {
			for (int col = 0; col < m_cubemapResolution; col++) {
				pixelDirection3d.x = start.x + ((float)col * 2.0f + 0.5f) / (float)m_cubemapResolution * right.x + ((float)row * 2.0f + 0.5f) / (float)m_cubemapResolution * up.x;
				pixelDirection3d.y = start.y + ((float)col * 2.0f + 0.5f) / (float)m_cubemapResolution * right.y + ((float)row * 2.0f + 0.5f) / (float)m_cubemapResolution * up.y;
				pixelDirection3d.z = start.z + ((float)col * 2.0f + 0.5f) / (float)m_cubemapResolution * right.z + ((float)row * 2.0f + 0.5f) / (float)m_cubemapResolution * up.z;

				float azimuth   = std::atan2f(pixelDirection3d.x, -pixelDirection3d.z) + (float)M_PI; // add pi to move range to 0-360 deg
				float elevation = std::atanf(pixelDirection3d.y / std::sqrtf(pixelDirection3d.x * pixelDirection3d.x + pixelDirection3d.z * pixelDirection3d.z)) + (float)M_PI / 2.0f;

				float colHdri = (azimuth / (float)M_PI / 2.0f) * m_width; // add pi to azimuth to move range to 0-360 deg
				float rowHdri = (elevation / (float)M_PI) * m_height;

				if (!m_filterLinear) {
					int colNearest = std::clamp((int)colHdri, 0, m_width - 1);
					int rowNearest = std::clamp((int)rowHdri, 0, m_height - 1);

					face[col * m_channels + m_cubemapResolution * row * m_channels]     = m_imageData[colNearest * m_channels + m_width * rowNearest * m_channels];     // red
					face[col * m_channels + m_cubemapResolution * row * m_channels + 1] = m_imageData[colNearest * m_channels + m_width * rowNearest * m_channels + 1]; // green
					face[col * m_channels + m_cubemapResolution * row * m_channels + 2] = m_imageData[colNearest * m_channels + m_width * rowNearest * m_channels + 2]; // blue
					if (m_channels > 3)
						face[col * m_channels + m_cubemapResolution * row * m_channels + 3] = m_imageData[colNearest * m_channels + m_width * rowNearest * m_channels + 3]; // alpha
				} else                                                                                                                                                      // perform bilinear interpolation
				{
					float intCol, intRow;
					float factorCol = std::modf(colHdri - 0.5f, &intCol); // factor gives the contribution of the next column, while the contribution of intCol is 1 - factor
					float factorRow = std::modf(rowHdri - 0.5f, &intRow);

					int low_idx_row    = static_cast<int>(intRow);
					int low_idx_column = static_cast<int>(intCol);
					int high_idx_column;
					if (factorCol < 0.0f) // modf can only give a negative value if the azimuth falls in the first pixel, left of the center, so we have to mix with the pixel on the opposite side of the panoramic image
						high_idx_column = m_width - 1;
					else if (low_idx_column == m_width - 1) // if we are in the right-most pixel, and fall right of the center, mix with the left-most pixel
						high_idx_column = 0;
					else
						high_idx_column = low_idx_column + 1;

					int high_idx_row;
					if (factorRow < 0.0f)
						high_idx_row = m_height - 1;
					else if (low_idx_row == m_height - 1)
						high_idx_row = 0;
					else
						high_idx_row = low_idx_row + 1;

					factorCol = std::abs(factorCol);
					factorRow = std::abs(factorRow);
					float f1  = (1 - factorRow) * (1 - factorCol);
					float f2  = factorRow * (1 - factorCol);
					float f3  = (1 - factorRow) * factorCol;
					float f4  = factorRow * factorCol;

					for (int j = 0; j < m_channels; j++) {
						unsigned char interpolatedValue                                     = static_cast<unsigned char>(m_imageData[low_idx_column * m_channels + m_width * low_idx_row * m_channels + j] * f1 + m_imageData[low_idx_column * m_channels + m_width * high_idx_row * m_channels + j] * f2 + m_imageData[high_idx_column * m_channels + m_width * low_idx_row * m_channels + j] * f3 + m_imageData[high_idx_column * m_channels + m_width * high_idx_row * m_channels + j] * f4);
						face[col * m_channels + m_cubemapResolution * row * m_channels + j] = std::clamp(interpolatedValue, (std::uint8_t)0, (std::uint8_t)255);
					}
				}
			}
		}
	}
}
