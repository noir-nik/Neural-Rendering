export module NeuralGraphics:Core.Types;
import std;

export namespace Types {

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using f32 = float;
using f64 = double;
} // namespace Types

export {
	using Types::f32;
	using Types::f64;
	using Types::i16;
	using Types::i32;
	using Types::i64;
	using Types::i8;
	using Types::u16;
	using Types::u32;
	using Types::u64;
	using Types::u8;
}
