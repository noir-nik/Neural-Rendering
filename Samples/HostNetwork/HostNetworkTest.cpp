import NeuralGraphics;
import std;

int main(int argc, char *argv[]) {
	HostNetwork hostNetwork2({{{
		Linear(),
		Relu(),
		Linear()
	}}});
}