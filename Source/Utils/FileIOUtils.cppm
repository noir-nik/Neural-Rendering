export module NeuralGraphics:FileIOUtils;
import std;

export namespace Utils {

using BinDataView   = std::span<std::byte const>;
using BinData       = std::vector<std::byte>;
using BinDataOption = std::optional<BinData>;

[[nodiscard]] auto ReadFile(std::string_view filename) -> std::optional<std::string>;
[[nodiscard]] auto ReadBinaryFile(std::string_view filename) -> std::optional<std::vector<std::byte>>;

} // namespace Utils
