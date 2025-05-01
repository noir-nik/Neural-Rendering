export module NeuralGraphics:FileIOUtils;
import std;

export namespace Utils {

[[nodiscard]] auto ReadFile(std::string_view filename) -> std::optional<std::string>;
[[nodiscard]] auto ReadBinaryFile(std::string_view filename) -> std::optional<std::vector<std::byte>>;

} // namespace Utils
