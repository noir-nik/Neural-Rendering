module;

module BRDFSample;

constexpr bool is_digit(char c) {
	return c >= '0' && c <= '9';
}

template <int N>
struct GetNumResult {
	std::array<u32, N> numbers{};
	std::size_t        count{};
};

constexpr bool contains17(std::string_view haystack, std::string_view needle) {
	if (needle.empty()) return true;
	for (std::size_t i = 0; i <= haystack.size() - needle.size(); ++i) {
		if (haystack.substr(i, needle.size()) == needle) {
			return true;
		}
	}
	return false;
}

constexpr bool contains(std::string_view haystack, std::string_view needle) {
	return haystack.find(needle) != std::string_view::npos;
}

static_assert(contains("hello world", " worl"));

template <std::size_t N>
constexpr auto get_last_numbers(std::string_view str) -> GetNumResult<N> {

	std::array<u32, N> result{};

	std::size_t count = 0;
	std::size_t i     = str.size();

	while (i > 0 && count < N) {
		if (is_digit(str[i - 1])) {
			std::size_t end = i;
			while (i > 0 && is_digit(str[i - 1])) {
				i--;
			}
			std::size_t start = i;

			int current_number = 0;
			int multiplier     = 1;
			for (std::size_t j = end; j > start; j--) {
				current_number += (str[j - 1] - '0') * multiplier;
				multiplier *= 10;
			}

			result[N - 1 - count] = current_number;
			count++;
		} else {
			i--;
		}
	}

	return {result, count};
}

using u32 = std::uint32_t;

struct FastKanParamsResult {
	u32 learnable;
	u32 total;
};

constexpr FastKanParamsResult count_fastkan_params(u32 input, u32 output, u32 layer_count, u32 layer_size, u32 grids) {

	u32 total     = 0;
	u32 learnable = 0;

	u32 current_in  = input;
	u32 current_out = layer_size;
	for (u32 i = 0; i <= layer_count; ++i) {
		if (i == layer_count) {
			current_out = output;
		}
		const u32 rbf_grid      = grids;
		const u32 rbf_denom_inv = 1;
		const u32 spline_weight = current_in * current_out * grids;
		const u32 base_weight   = current_in * current_out;
		const u32 base_bias     = current_out;

		total += rbf_grid + rbf_denom_inv + spline_weight + base_weight + base_bias;
		learnable += spline_weight + base_weight + base_bias;
		current_in = layer_size;
	}

	return {learnable, total};
}

constexpr u32 count_nbrdf_params(u32 input, u32 output, u32 hidden_features, u32 layers) {
	u32 result = 0;

	if (layers > 0) {
		result += input * hidden_features + hidden_features;
		for (u32 i = 0; i < layers - 1; ++i) {
			result += hidden_features * hidden_features + hidden_features;
		}
		result += hidden_features * output + output;
	} else {
		result += input * output + output;
	}

	return result;
}

constexpr auto parse_model = [](SV name) constexpr -> BRDFModelData {
	auto const numbers = get_last_numbers<3>(name).numbers;
	// return {name, numbers[0], numbers[1], numbers[2]};
	u32 learnable_params, total_params;

	u32              input  = 6;
	u32              output = 3;
	std::string_view type;
	if (contains(name, "KAN")) {
		type             = "KAN";
		auto res         = count_fastkan_params(input, output, numbers[0], numbers[1], numbers[2]);
		learnable_params = res.learnable;
		total_params     = res.total;
	} else if (contains(name, "NBRDF")) {
		type             = "NBRDF";
		learnable_params = total_params = count_nbrdf_params(input, output, numbers[2], numbers[1]);
	}

	return {
		.name             = name,
		.type             = type,
		.layers           = numbers[0],
		.layer_size       = numbers[1],
		.grids            = numbers[2],
		.learnable_params = learnable_params,
		.total_params     = total_params,
	};
};

static constexpr auto gBRDFModelNames = std::array{
#define BRDF_NAME(x) parse_model(#x),
#include "BRDFModels.def"
#undef BRDF_NAME
};

auto BRDFSample::GeneratedNames() -> CSpan<BRDFModelData> {
	return gBRDFModelNames;
};
