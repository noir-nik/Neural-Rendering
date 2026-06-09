module;
// #include <cmath>
module HdriToCubemap;

import std;
import StbModule;

void HdriToCubemapBase::write_cubemap(std::string_view const output_folder, std::span<void const* const> faces) {
	stbi_flip_vertically_on_write(1);

	auto filenames = std::array{"front", "back", "left", "right", "up", "down"};

	auto path_array = std::array<char, 512>{};

	for (int i = 0; i < 6; i++) {
		int success;

		std::snprintf(
			std::data(path_array), std::size(path_array),
			"%*s%s%s%s",
			static_cast<int>(std::size(output_folder)), std::data(output_folder),
			(output_folder.empty() ? "" : "/"),
			filenames[i], get_is_hdri() ? ".hdr" : ".png");

		if (get_is_hdri()) {
			success = stbi_write_hdr(std::data(path_array), get_resolution(), get_resolution(), get_num_channels(), (const float*)faces[i]);
		} else {
			success = stbi_write_png(std::data(path_array), get_resolution(), get_resolution(), get_num_channels(), (const unsigned char*)faces[i], 0);
		}
		if (!success)
			std::printf("Warning: could not write \"%s\"", std::data(path_array));
	}
}
