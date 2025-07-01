
module LoadWeights;
import WeightsLoader;

int load_weights(std::string_view filename, std::vector<LayerVariant>& layers, std::vector<float>& h_params, std::string_view header) {

	WeightsLoader loader;
	if (!loader.Init(filename, header)) {
		std::printf("Failed to load weights from %s\n", filename.data());
		return 1;
	}

	h_params.clear();
	h_params.resize(loader.GetFileSize() / sizeof(float));
	layers.clear();
	layers.reserve(50);

	float*   src_weights = h_params.data();
	unsigned layer_count = 0;
	while (loader.HasNext()) {
		u32 const rows = loader.NextRows();
		u32 const cols = loader.NextCols();
		layers.push_back(Linear(cols, rows));
		// layers.push_back(Activation(ActivationKind::ReluFwd));

		// std::printf("%u, input: %u, output: %u\n", layer_count++, cols, rows);

		std::size_t const weights_count = rows * cols;
		std::size_t const biases_count  = rows;

		bool loaded = loader.LoadNext(src_weights, src_weights + weights_count);
		if (!loaded) {
			std::printf("Error loading weights\n");
			std::exit(1);
		}
		src_weights += weights_count + biases_count;
	}

	return {};
}
