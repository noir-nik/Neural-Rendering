module CubeMap;
import StbModule;
#define LIFT(fn) [&](auto&&... args) { return fn(args...); }

using u32 = std::uint32_t;

auto CubeMetadata::destroy() -> void {
	if (!is_valid()) return;

	for (auto im : cubemap) {
		stbi_image_free(static_cast<void*>(im));
	}
	set_invalid();
}

auto CubeMetadata::set_invalid() -> void {
	std::ranges::for_each(cubemap, [](auto&& c) { c = nullptr; });
	width = height = channels = {};
}

auto CubeMetadata::is_valid() -> bool {
	return std::ranges::all_of(cubemap, LIFT(bool));
}

auto LoadCubemap(std::string_view env_map_folder_path) -> CubeMetadata {
	// Load all the textures.

	int width{0};
	int height{0};
	int channels{0};

	char buf[512];

	auto concat_str = [&](std::string_view a, std::string_view b) -> std::string_view {
		auto const result = std::snprintf(buf, sizeof(buf), "%s/%s", a.data(), b.data());
		return {buf, static_cast<std::size_t>(result)};
	};

	auto loads = [&](std::string_view filename) -> unsigned char* {
		auto const path     = concat_str(env_map_folder_path, filename);
		std::array old_data = {width, height, channels};

		auto res = stbi_load(path.data(), &width, &height, &channels, 4);

		if (!res) {
			std::printf("[WARNING] Failed to load file %s\n", path.data());
			return res;
		}
		// verify no difference old_data - new_data
		auto [old_width, old_height, old_channels] = old_data;
		{
			std::array new_data   = {width, height, channels};
			std::array data_names = {"width", "height", "channels"};
			auto       get_data   = [&](u32 i) { return std::tuple{old_data[i], new_data[i], data_names[i]}; };
			if (std::all_of(old_data.begin(), old_data.end(), [](auto const& i) { return i != 0; })) {

				bool const skip_check_channels = true;
				for (u32 i = 0; i < old_data.size() - (skip_check_channels * 1); ++i) {
					auto [old, new_, name] = get_data(i);
					if (old_data[i] != new_data[i]) {
						std::printf("Cubemap size mismatch %s Old: %u, New: %u\n", name, old, new_);
						// std::exit(1);
					}
				}
			}
		}
		return res;
	};

	auto cube_data = std::array{
		loads("nx.png"),
		loads("px.png"),
		loads("py.png"),
		loads("ny.png"),
		loads("nz.png"),
		loads("pz.png"),
	};

	CubeMetadata cube{
		.cubemap = std::array{
			loads("nx.png"),
			loads("px.png"),
			loads("py.png"),
			loads("ny.png"),
			loads("nz.png"),
			loads("pz.png"),
		},
		.width    = width,
		.height   = height,
		.channels = channels,
	};

	if (!cube.is_valid()) {
		cube.destroy();
		return {};
	}
	return {cube_data, width, height, channels};
}
