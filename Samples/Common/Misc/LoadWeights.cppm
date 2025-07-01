
export module LoadWeights;

// import LayerInfo;
import NeuralGraphics;
import std;

export [[nodiscard]]
int load_weights(std::string_view filename, std::vector<LayerVariant>& layers, std::vector<float>& h_params, std::string_view header = "");
