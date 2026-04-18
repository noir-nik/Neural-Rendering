export module NeuralGraphics:MakeString;
import std;

export namespace Utils {

template <typename... Args>
auto make_string(Args&&... args)
	requires(std::is_convertible_v<Args, std::string_view> && ...)
{
	using SV = std::string_view;

	std::size_t total_size = 0;
	((total_size += SV(std::forward<Args>(args)).size()), ...);
	std::string result;
	result.resize(total_size);

	std::size_t offset = 0;
	((std::copy(
		  SV(std::forward<Args>(args)).cbegin(),
		  SV(std::forward<Args>(args)).cend(),
		  result.begin() + offset),
	  offset += SV(std::forward<Args>(args)).size()),
	 ...);
	return result;
}

} // namespace Utils
