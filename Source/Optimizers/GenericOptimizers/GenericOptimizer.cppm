export module NeuralGraphics:GenericOptimizer;
import :Core;
import std;

export namespace ng {

class GenericOptimizer {
public:
	virtual ~GenericOptimizer() {}
	virtual void Step() = 0;
};

} // namespace ng
