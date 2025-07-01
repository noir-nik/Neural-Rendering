module;

module WeightsLoader;
 
import std;

bool WeightsLoader::Init(std::string_view filename, std::string_view header) {
	constexpr auto kMaxHeaderSize = 128u;

	file.open(filename.data(), std::ios::ate | std::ios::binary);
	file_size = static_cast<std::size_t>(file.tellg());
	file.seekg(0);

	// Check header
	auto header_size = std::size(header);
	if (header_size > 0) {
		if (header_size > kMaxHeaderSize) return 0;
		if (file_size < header_size) return 0;
		char buf[kMaxHeaderSize];
		file.read(buf, header_size);
		buf[header_size] = '\0';

		if (std::string_view(header) != buf) return 0;
	}

	file.read(reinterpret_cast<char*>(&layers), sizeof(u32));
	if (!file.good()) {
		layers = 0;

		file.close();

		return 0;
	} else {
		ReadNextLayerInfo();
		return 1;
	}
}

bool WeightsLoader::LoadNext(float* weights, float* bias) {
	if (layers == 0) return false;

	const u32 count = rows * cols;
	if (count > 0) {
		file.read(reinterpret_cast<char*>(weights), count * sizeof(float));
		file.read(reinterpret_cast<char*>(bias), rows * sizeof(float));
	}

	if (!file) {
		layers = 0;
		rows = cols = 0;
		return false;
	}

	layers -= 1;
	ReadNextLayerInfo();
	return true;
}

void WeightsLoader::ReadNextLayerInfo() {
	if (layers == 0) {
		rows = cols = 0;
		return;
	}

	file.read(reinterpret_cast<char*>(&rows), sizeof(u32));
	file.read(reinterpret_cast<char*>(&cols), sizeof(u32));
	if (!file.good()) {
		layers = 0;
	}
}
