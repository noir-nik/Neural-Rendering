module FastKANCoopVec;

import std;

import nlohmann.json;

using json = nlohmann::json;
using Utils::make_string;

[[nodiscard]] auto ReadKANWeights(std::string_view file_name) -> ReadKANResult {
	using namespace nlohmann;

	std::string prefix = std::string(file_name);
	// std::string prefix = (network_info.file_name);
	if (prefix.ends_with(".json")) {
		prefix = prefix.substr(0, prefix.size() - 5);
	} else if (prefix.ends_with(".bin")) {
		prefix = prefix.substr(0, prefix.size() - 4);
	}

	auto json_name = prefix + ".json";

	using Utils::make_string;

	std::ifstream json_file(json_name);
	if (!json_file.is_open()) {
		// std::cerr << "Error: Could not open " << prefix << ".json" << std::endl;
		// std::exit(1);
		return std::unexpected(make_string("Could not open ", json_name));
	}

	json json_manifest;
	json_file >> json_manifest;

	std::string dtype_str = json_manifest["dtype"];
	enum class DataType {
		Float32,
		Float16,
	};
	DataType dtype;
	if (dtype_str == "float32") {
		dtype = DataType::Float32;
	} else {
		// std::cerr << "Error: Unsupported data type " << dtype_str << std::endl;
		// std::exit(1);
		// return std::unexpected(std::string("Unsupported data type ") + dtype_str);
		return std::unexpected(make_string("Unsupported data type ", dtype_str));
	}

	std::ifstream bin_file(prefix + ".bin", std::ios::binary);
	if (!bin_file.is_open()) {
		// std::cerr << "Error: Could not open " << prefix << ".bin" << std::endl;
		// std::exit(1);
		return std::unexpected(make_string("Could not open ", prefix, ".bin"));
	}

	std::vector<float> bin_data; //((std::istreambuf_iterator<std::streambuf>(bin_file)), std::istreambuf_iterator<std::streambuf>());
	bin_file.seekg(0, std::ios::end);
	bin_data.resize(std::streamsize(bin_file.tellg()));
	bin_file.seekg(0);
	bin_file.read(reinterpret_cast<char*>(bin_data.data()), bin_data.size());
	bin_file.close();

	FastKan kan;

	auto const& json_layers = json_manifest["layers"];
	kan.layers_.reserve(json_layers.size());

	for (const auto& json_layer : json_layers) {
		FastKanLayer kan_layer;
		for (const auto& param : json_layer["params"].items()) {
			auto const& name   = param.key(); // param.first;
			auto const& info   = param.value();
			u32         offset = info["offset"];
			u32         size   = info["size"];
			auto        shape  = info["shape"];

			// auto start = bin_data.data() + offset;

			auto ldata = KANBuffer(offset, size);

			// shape
			// for (u32 i = 0; i < shape.size(); ++i) {
			// 	ldata.shape_[ldata.shape_size_++] = shape[i];
			// }

			ldata.set_shape(shape);

			if (name == "rbf_grid") {
				kan_layer.rbf_grid() = ldata;
			} else if (name == "rbf_denom_inv") {
				kan_layer.rbf_denom_inv() = ldata;
			} else if (name == "spline_weight") {
				kan_layer.spline_weight() = ldata;
			} else if (name == "base_weight") {
				kan_layer.base_weight() = ldata;
			} else if (name == "base_bias") {
				kan_layer.base_bias() = ldata;
			}
		}
		kan.layers_.push_back(kan_layer);
	}
	kan.buffer_ = std::move(bin_data);

	// return {std::move(json_manifest), std::move(layers)};
	return kan;
}

