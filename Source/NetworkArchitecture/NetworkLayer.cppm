export module NeuralGraphics:Network.Layer;
import std;

export
class Linear {
};

export
class Relu {
};

export
using NetworkLayerVariant = std::variant<Linear, Relu>;
