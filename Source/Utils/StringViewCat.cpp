module NeuralGraphics;
import :StringViewCat;
import std;

static constexpr std::string_view hello = "Hello, ";
static constexpr std::string_view world = "World";
static constexpr std::string_view bang  = "!";

using Cat = Utils::StringViewCat<hello, world, bang>;
static_assert(Cat::value == "Hello, World!");
