module NeuralGraphics;
import :FileIOUtils;
import std;

namespace ng::inline Utils {

auto ReadFile(std::string_view filename) -> std::optional<std::string> {
	if (auto file = std::ifstream(filename.data(), std::ios::ate)) {
		std::string content;
		content.reserve(static_cast<std::size_t>(file.tellg()));
		file.seekg(0);
		content.assign((std::istreambuf_iterator<std::string::value_type>(file)),
					   std::istreambuf_iterator<std::string::value_type>());
		file.close();
		return std::move(content);
	}
	return std::nullopt;
}

auto ReadBinaryFile(std::string_view filename) -> std::optional<std::vector<std::byte>> {
	std::ifstream file(filename.data(), std::ios::ate | std::ios::binary);
	if (!file.is_open()) {
		return std::nullopt;
	}
	std::size_t            file_size = static_cast<std::size_t>(file.tellg());
	std::vector<std::byte> buffer(file_size);
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), file_size);
	file.close();
	return std::move(buffer);
}
} // namespace ng::Utils
