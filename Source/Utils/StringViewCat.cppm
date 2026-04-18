export module NeuralGraphics:StringViewCat;
import std;

export namespace Utils {

// template <class... Svs>
// 	requires(std::is_convertible_v<std::remove_cvref_t<Svs>, std::string_view> && ...)
template <std::string_view const&... Svs>
struct StringViewCat {

	// template <auto... ts>
	// StringViewCat<ts...>(auto&&... s);

	static constexpr std::size_t total_size = (Svs.size() + ...);

	static constexpr auto get() {
		std::array<char, total_size + 1> arr{};

		std::size_t offset = 0;

		auto append = [&](std::string_view const sv) {
			for (auto c : sv)
				arr[offset++] = c;
		};

		(append(Svs), ...);
		arr[offset] = '\0';
		return arr;
	}

	static constexpr std::array<char, total_size + 1> arr = get();
	static constexpr std::string_view                 value{arr.data(), total_size};
};

// template <typename... Args>
// 	requires(std::is_convertible_v<Args, std::string_view> && ...)
// StringViewCat(Args&&... args) -> StringViewCat<Args...>;

// template <auto... s>
// template <std::string_view const&... Svs>
// StringViewCat(auto&&... s) -> StringViewCat<std::string_view(ts)...>;

template <auto...>
StringViewCat(auto&&... ts) -> StringViewCat<std::string_view(ts)...>;

} // namespace Utils

static constexpr std::string_view hello = "Hello, ";
static constexpr std::string_view world = "World";
static constexpr std::string_view bang  = "!";

using Cat = Utils::StringViewCat<hello, world, bang>;

static_assert(Cat::value == "Hello, World!");

// static_assert(Utils::StringViewCat { hello, world, bang } ::value == "Hello, World!");
