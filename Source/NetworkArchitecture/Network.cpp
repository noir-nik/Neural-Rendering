module NeuralGraphics;
import :Network;
import std;

Network::Network(std::span<NetworkLayerVariant const> layers) {
	this->layers.assign(layers.begin(), layers.end());
}