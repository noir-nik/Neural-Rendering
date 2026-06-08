module BRDFSample;
#include "Log.h"

template <typename Range, typename Proj = std::identity>
constexpr inline auto contains(Range&& range, auto&& value, Proj&& proj = std::identity{}) {
	for (auto&& v : range)
		if (std::invoke(proj, v) == value)
			return true;
	return false;
};


void BRDFSample::RunBenchmark(TestOptions const& options) {
	LOG_DEBUG("BRDFSample::RunBenchmark()");
	struct TestData {
		vk::Pipeline                pipeline;
		VulkanCoopVecNetwork const* network;
	};
	// WindowManager::WaitEvents();
	// if (window.GetShouldClose()) return;

	// window.SetWindowMode(WindowMode::eFullscreen);
	auto [width, height] = options.resolution;
	window.SetSize(width, height);
	// window.Hide();
	// std::printf("Resizing to %dx%d\n", width, height);
	RecreateSwapchain(width, height);

	constexpr u32 kTestRunsCount = 64;

	// constexpr u32 kMaxTestKinds = std::to_underlying(BrdfFunctionType::eCount);
	const u32 kMaxTestKinds = pipelines_header.size();

	constexpr u32 kMaxTests = 64;
	// std::vector<std::array<u64, kMaxTestKinds>> test_times(kTestRunsCount);
	std::array<std::array<u64, kMaxTests>, kTestRunsCount> test_times;

	int first_test{}, last_test = kMaxTestKinds;

	if (benchmark_single) {
		first_test = std::to_underlying(function_type);
		last_test  = first_test + 1;
	} else {
		first_test = *function_id;
		last_test  = first_test + 1;
	}

	bool is_header = true;
	// bool is_header = false;

	if (is_header) {
		first_test = 0;
		last_test  = kMaxTestKinds;
		// last_test  = 5;
	}

	// std::printf("Running %d tests\n", last_test - first_test);
	// std::printf("test id: %d\n", first_test);

	// BrdfFunctionType skip[] = {BrdfFunctionType::eWeightsInHeader};
	BrdfFunctionType skip[] = {};

	// std::mem_fn(&BRDFSample::DrawWindow);
	LOG_DEBUG("BRDFSample::DrawWindow()");

	auto draw = [&](u32 id) {
		if (is_header) {
			return DrawWindow(pipelines_header[id]);
		} else {
			return DrawWindow(pipelines[id]);
		};
	};

	// Warm up gpu clocks
	constexpr u32 kWarmupCount = 2;
	for (u32 t_i = first_test; t_i < last_test; ++t_i) {
		if (contains(skip, BrdfFunctionType(t_i))) continue;
		for (u32 iter = 0; iter < kWarmupCount; ++iter) {
			(void)draw(t_i);
		}
	}

	bool with_in_header = false;

	for (u32 t_i = first_test; t_i < last_test; ++t_i) {
		if (contains(skip, BrdfFunctionType(t_i))) continue;
		for (u32 iter = 0; iter < kTestRunsCount; ++iter) {
			// WindowManager::PollEvents();
			u64   time_nanoseconds = draw(t_i);
			float ns_per_tick      = physical_device.GetNsPerTick();
			float elapsed_ms       = (time_nanoseconds * ns_per_tick) / 1e6f;
			test_times[iter][t_i]  = time_nanoseconds;
		}
	}

	char const* names[] = {"Classic", "CoopVec", "WeightsInBuffer", "WeightsInBufferFloat16", "WeightsInHeader", "Kan"};

	// Print csv
	// std::printf("Print csv\n");
	// std::printf("Classic,CoopVec,WeightsInBuffer,WeightsInBufferFloat16,WeightsInHeader\n");

	char const* header_names[] = {
#define BRDF_NAME(x) #x,
// #include "SINEKAN_HeaderNames.def"
#include "HeaderNames.def"
		// #include "CHEBYKAN_HeaderNames.def"
		// #include "RELUKAN_HeaderNames.def"
	};

	// for (u32 t_i = first_test; t_i < last_test; ++t_i) {
	// 	if (contains(skip, BrdfFunctionType(t_i))) continue;
	// 	if (is_header) {
	// 		std::printf("%s", header_names[t_i]);
	// 	} else {
	// 		// std::printf("t_i %u", t_i);
	// 		std::printf("%s", names[t_i]);
	// 	}
	// 	if (t_i < last_test - 1 && !contains(skip, BrdfFunctionType(t_i + 1))) std::printf(",");
	// }
	// std::printf("\n");

	// // std::printf("Print times\n");
	// for (u32 iter = 0; iter < kTestRunsCount; ++iter) {
	// 	auto const& tests_row = test_times[iter];
	// 	// print with ,
	// 	for (u32 t_i = first_test; t_i < last_test; ++t_i) {
	// 		if (contains(skip, BrdfFunctionType(t_i))) continue;
	// 		std::printf("%llu", tests_row[t_i]);
	// 		if (t_i < last_test - 1 && !contains(skip, BrdfFunctionType(t_i + 1))) std::printf(",");
	// 	}
	// 	std::printf("\n");
	// }

	auto print_name =
		// false
		true;
	if (print_name) {
		for (u32 t_i = first_test; t_i < last_test; ++t_i) {
			if (contains(skip, BrdfFunctionType(t_i))) continue;
			if (is_header) {
				std::printf("%s", header_names[t_i]);
			} else {
				std::printf("%s", names[t_i]);
			}
			if (t_i < last_test - 1 && !contains(skip, BrdfFunctionType(t_i + 1))) std::printf(",");
		}
		std::printf("\n");
	}

	auto print_all =
		// false
		true;
	if (print_all)
		for (u32 iter = 0; iter < kTestRunsCount; ++iter) {
			auto const& tests_row = test_times[iter];
			// print with ,
			for (u32 t_i = first_test; t_i < last_test; ++t_i) {
				if (contains(skip, BrdfFunctionType(t_i))) continue;
				std::printf("%llu", tests_row[t_i]);
				if (t_i < last_test - 1 && !contains(skip, BrdfFunctionType(t_i + 1))) std::printf(",");
			}
			std::printf("\n");
		}

	// print means
	// std::printf("Classic,CoopVec,WeightsInBuffer,WeightsInBufferFloat16,WeightsInHeader\n");
	u32  first_row = 3;
	u64  ms_in_ns  = 1e6;
	auto print_means =
		// true
		false;
	if (print_means) {
		for (u32 t_i = first_test; t_i < last_test; ++t_i) {
			if (contains(skip, BrdfFunctionType(t_i))) continue;
			double mean = 0;
			for (u32 iter = first_row; iter < kTestRunsCount; ++iter) {
				mean += double(test_times[iter][t_i]) / ms_in_ns;
			}
			mean /= (kTestRunsCount - first_row);
			std::printf("%f", mean);
			if (t_i < last_test - 1 && !contains(skip, BrdfFunctionType(t_i + 1))) std::printf(",");
			// if (t_i > 0  and (t_i + 1) % 4 == 0) std::printf("\n");
		}
		std::printf("\n");
	}
}

auto BRDFSample::ParseArgs(int argc, char const* argv[]) -> char const* {
	LOG_DEBUG("BRDFSample::ParseArgs()");
	auto args_range = std::span(argv + 1, argc - 1);

	if (std::ranges::contains(args_range, std::string_view("--help")))
		return "--help";

	for (auto it = args_range.begin(); it != args_range.end(); ++it) {
		auto arg = std::string_view(*it);
		if (arg == "--benchmark" || arg == "-b") is_test_mode = true;
		else if (arg == "--verbose" || arg == "-v") verbose = true;
		else if (arg == "--validation") use_validation = true;
		else if (arg == "--kind") {
			if ((it + 1) == args_range.end()) return "expected <kind>";
			auto kind = std::string_view(*(it + 1));
			int  value;
			if (std::from_chars(kind.data(), kind.data() + kind.size(), value).ec != std::errc()) return *(it + 1);
			if (value < 0 || value >= std::to_underlying(BrdfFunctionType::eCount)) return *(it + 1);
			function_type    = static_cast<BrdfFunctionType>(value);
			benchmark_single = true;
			++it;
		} else if (arg == "-f") {
			if ((it + 1) == args_range.end()) return "expected <id>";
			auto str = std::string_view(*(it + 1));
			int  value;
			if (std::from_chars(str.data(), str.data() + str.size(), value).ec != std::errc()) return *(it + 1);
			// if (value < 0 || value >= kTestFunctionsCount) return *(it + 1);
			// function_type    = BrdfFunctionType::eWeightsInHeader;
			benchmark_single = true;
			function_id      = value;
			++it;
		} else if (arg == "-w") { // input weights file
			if ((it + 1) == args_range.end()) return "expected <file>";
			auto str = std::string_view(*(it + 1));
			// if (!std::filesystem::exists(str)) return *(it + 1);
			weights_file_name = str;
			++it;
		} else if (arg == "-wh") {
			if ((it + 1) == args_range.end()) return "expected <file>";
			auto str = std::string_view(*(it + 1));
			// if (!std::filesystem::exists(str)) return *(it + 1);
			this->header = str;
			++it;
		} else if (arg == "-kw") { // input weights file
			if ((it + 1) == args_range.end()) return "expected <file>";
			auto str = std::string_view(*(it + 1));

			kan_weights_file_name = str;
			++it;
		} else if (arg == "-cm") { // input cubemap folder
			if ((it + 1) == args_range.end()) return "expected <folder>";
			auto str = std::string_view(*(it + 1));

			cubemap_folder_path = str;
			++it;
		} else if (arg == "-obj") { // input weights file
			if ((it + 1) == args_range.end()) return "expected <file>";
			auto str = std::string_view(*(it + 1));

			obj_path = str;
			++it;
		} else if (arg == "-fv") {
			if ((it + 1) == args_range.end()) return "expected <id>";
			auto str = std::string_view(*(it + 1));
			int  value;
			if (std::from_chars(str.data(), str.data() + str.size(), value).ec != std::errc()) return *(it + 1);
			if (value < kMinFastKANVersion || value > kMaxFastKANVersion) return *(it + 1);
			fastkan_version = value;
			++it;
		} else return *it;
	}

	return nullptr;
}

auto PrintUsage([[maybe_unused]] int argc, char const* argv[]) -> void {
	std::printf("Usage: %s [--help] [--benchmark | -b] [--verbose | -v] [--validation] [--kind <kind>]\n",
				std::filesystem::path(argv[0]).filename().string().c_str());
	std::printf("  --kind <kind>\n");
	std::printf("      Kind of BRDF function to run:\n");
	std::printf("        0: Classic\n");
	std::printf("        1: Coop Vec (Default)\n");
	std::printf("        2: Weights in buffer\n");
	std::printf("        3: Weights in buffer float16\n");
	std::printf("        4: Weights in header\n");
	std::printf("        5: Kan\n");
	std::printf("  --benchmark | -b\n");
	std::printf("      Run benchmark\n");
	std::printf("  --verbose | -v\n");
	std::printf("      Run benchmark with verbose output\n");
	std::printf("  --validation\n");
	std::printf("      Enable validation\n");
	std::printf("  -fv, --fastkan-version\n");
	std::printf("      FastKAN version (int)\n");
	std::printf("\n");
};

extern "C++" auto main(int argc, char const* argv[]) -> int {
	std::filesystem::current_path(std::filesystem::absolute(argv[0]).parent_path());
	BRDFSample sample;

	if (char const* unknown_arg = sample.ParseArgs(argc, argv); unknown_arg) {
		if (unknown_arg != std::string_view("--help"))
			std::printf("Error in argument: %s\n", unknown_arg);
		PrintUsage(argc, argv);
		return 0;
	}

	TestOptions options{
		.resolution = {640, 480},
		.test_count = 64,
	};

	int2 res_arr[] = {
		{1920, 1080},
		// {3840, 2160},
		// {512, 512},
		{640, 480},
		{1280, 720},
		{1920, 1080},
		{2560, 1440},
		// {3840, 2160},
	};

	auto res_count = //

		// std::size(res_arr);
		1;

	sample.Init();
	if (sample.IsTestMode()) {
		for (int i = 0; i < res_count; ++i) {
			options.resolution = res_arr[i];
			// std::printf("resolution: %d x %d\n", res_arr[i].x, res_arr[i].y);
			sample.RunBenchmark(options);
		}
	} else {
		sample.Run();
	}
	return 0;
}
