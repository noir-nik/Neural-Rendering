export module CubeMap;

import std;

export enum class CubeType {
	U32,
	F32,
};

export struct CubeMetadata {
	static constexpr std::size_t kNumFaces = 6;

	std::array<unsigned char*, kNumFaces> cubemap;

	// CubeType type{CubeType::F32};
	int      width{0};
	int      height{0};
	int      channels{0};

	auto image_size() const -> std::size_t { return width * height * channels; }
	auto image_size_bytes() const -> std::size_t { return image_size() * sizeof(*cubemap[0]); }

	auto size() const -> std::size_t { return image_size() * kNumFaces; }
	auto size_bytes() const -> std::size_t { return image_size_bytes() * kNumFaces; }
	auto images() const -> std::span<const unsigned char* const> { return cubemap; }

	auto is_valid() -> bool;
	auto destroy() -> void;

private:
	auto set_invalid() -> void;
};
export auto LoadCubemap(std::string_view env_map_folder_path) -> CubeMetadata;
