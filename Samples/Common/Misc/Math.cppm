export module Math;
import std;

export namespace math {

struct int2;
struct int3;
struct int4;
struct uint2;
struct uint3;
struct uint4;
struct float2;
struct float3;
struct float4;

struct quat;

struct uchar4;

struct float3x3;
struct float4x4;

using std::acosf;
using std::asinf;
using std::atan2f;
using std::atanf;
using std::cosf;
using std::sinf;
using std::tanf;

constexpr inline float DEG_TO_RAD = 3.14159265358979323846f / 180.0f;

typedef struct int2 {
	int x, y;
} int2;
typedef struct int3 {
	int x, y, z;
} int3;
typedef struct int4 {
	int x, y, z, w;
} int4;
typedef struct uint2 {
	unsigned int x, y;
} uint2;
typedef struct uint3 {
	unsigned int x, y, z;
} uint3;
typedef struct uint4 {
	unsigned int x, y, z, w;
} uint4;

typedef struct float2 vec2;
typedef struct float3 vec3;
typedef struct float4 vec4;

typedef struct float3x3 mat3;
typedef struct float4x4 mat4;

struct float2 {
	inline float2() : x(0.0f), y(0.0f) {}
	inline float2(float x, float y) : x(x), y(y) {}
	inline explicit float2(float val) : x(val), y(val) {}
	inline explicit float2(float const a[2]) : x(a[0]), y(a[1]) {}

	inline auto operator[](this auto&& self, int i) -> auto&& { return std::forward<decltype(self)>(self).M[i]; }

	union {
		struct {
			float x, y;
		};
		struct {
			float r, g;
		};
		float M[2];
	};
};

struct float3 {

	constexpr inline float3() : x(0.0f), y(0.0f), z(0.0f) {}
	constexpr inline float3(float x, float y, float z) : x(x), y(y), z(z) {}
	constexpr inline explicit float3(float val) : x(val), y(val), z(val) {}
	constexpr inline explicit float3(float const a[3]) : x(a[0]), y(a[1]), z(a[2]) {}

	constexpr inline auto operator[](int i) -> float& { return M[i]; }
	constexpr inline auto operator[](int i) const -> float const& { return M[i]; }

	constexpr inline auto begin() -> float* { return &x; }
	constexpr inline auto begin() const -> float const* { return &x; }
	constexpr inline auto end() -> float* { return &z + 1; }
	constexpr inline auto end() const -> float const* { return &z + 1; }

	// clang-format off
	union {
		struct { float x, y, z; };
		struct { float r, g, b; };
		float M[3];
	};
	// clang-format on
};

struct float4 {
	constexpr inline float4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
	constexpr inline float4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
	constexpr inline explicit float4(float val) : x(val), y(val), z(val), w(val) {}
	constexpr inline explicit float4(float const a[4]) : x(a[0]), y(a[1]), z(a[2]), w(a[3]) {}

	inline explicit float4(float2 a, float z = 0.0f, float w = 0.0f) : x(a.x), y(a.y), z(z), w(w) {}
	inline explicit float4(float2 a, float2 b) : x(a.x), y(a.y), z(b.x), w(b.y) {}
	inline explicit float4(float3 a, float w = 0.0f) : x(a.x), y(a.y), z(a.z), w(w) {}

	constexpr inline auto operator[](this auto&& self, int i) -> auto&& { return std::forward<decltype(self)>(self).M[i]; }

	// clang-format off
	union {
		struct { float x, y, z, w; };
		struct { float r, g, b, a; };
		struct { float2 xy, zw; };
		struct { float2 rg, ba; };
		struct { float3 xyz; };
		struct { float3 rgb; };
		float M[4];
	};
	// clang-format on
};

struct float3x3 {
	inline float3x3() { identity(); }

	inline void identity() {
		m_col[0] = float3{1.0f, 0.0f, 0.0f};
		m_col[1] = float3{0.0f, 1.0f, 0.0f};
		m_col[2] = float3{0.0f, 0.0f, 1.0f};
	}

	inline explicit float3x3(float a00, float a01, float a02,
							 float a10, float a11, float a12,
							 float a20, float a21, float a22) {
		m_col[0] = float3{a00, a01, a02};
		m_col[1] = float3{a10, a11, a12};
		m_col[2] = float3{a20, a21, a22};
	}

	inline float3x3(float4x4 const& from4x4);

	constexpr inline auto col(this auto&& self, int i) -> auto&& { return std::forward_like<decltype(self)>(self.operator[](i)); }
	constexpr inline auto operator[](this auto&& self, int col) -> auto&& { return std::forward<decltype(self)>(self).m_col[col]; }
	constexpr inline auto operator[](this auto&& self, int row, int col) -> auto&& { return std::forward<decltype(self)>(self).m_col[col][row]; }
	constexpr inline auto operator()(this auto&& self, int row, int col) -> auto&& { return std::forward<decltype(self)>(self).m_col[col][row]; }

	float3 m_col[3];
};

struct float4x4 {
	constexpr inline float4x4() { identity(); }

	constexpr inline explicit float4x4(float const A[16]) {
		m_col[0] = float4{A[0], A[4], A[8], A[12]};
		m_col[1] = float4{A[1], A[5], A[9], A[13]};
		m_col[2] = float4{A[2], A[6], A[10], A[14]};
		m_col[3] = float4{A[3], A[7], A[11], A[15]};
	}

	constexpr inline explicit float4x4(
		float a1, float a2, float a3, float a4,
		float a5, float a6, float a7, float a8,
		float a9, float a10, float a11, float a12,
		float a13, float a14, float a15, float a16) {
		m_col[0] = float4{a1, a5, a9, a13};
		m_col[1] = float4{a2, a6, a10, a14};
		m_col[2] = float4{a3, a7, a11, a15};
		m_col[3] = float4{a4, a8, a12, a16};
	}

	constexpr inline explicit float4x4(float3x3 const& from3x3) {
		m_col[0] = float4{from3x3[0][0], from3x3[0][1], from3x3[0][2], 0.0f};
		m_col[1] = float4{from3x3[1][0], from3x3[1][1], from3x3[1][2], 0.0f};
		m_col[2] = float4{from3x3[2][0], from3x3[2][1], from3x3[2][2], 0.0f};
		m_col[3] = float4{0.0f, 0.0f, 0.0f, 1.0f};
	}

	constexpr inline explicit float4x4(float4 const& col0, float4 const& col1, float4 const& col2, float4 const& col3) {
		m_col[0] = col0;
		m_col[1] = col1;
		m_col[2] = col2;
		m_col[3] = col3;
	}

	constexpr inline void identity() {
		m_col[0] = float4{1.0f, 0.0f, 0.0f, 0.0f};
		m_col[1] = float4{0.0f, 1.0f, 0.0f, 0.0f};
		m_col[2] = float4{0.0f, 0.0f, 1.0f, 0.0f};
		m_col[3] = float4{0.0f, 0.0f, 0.0f, 1.0f};
	}

	constexpr inline auto col(this auto&& self, int i) -> decltype(std::forward<decltype(self)>(self).m_col[i]) {
		return std::forward_like<decltype(self)>(self.operator[](i));
	}

	constexpr inline void set_col(int i, float4 const& a_col) { m_col[i] = a_col; }
	constexpr inline void set_row(int i, float4 const& value) {
		m_col[0][i] = value[0];
		m_col[1][i] = value[1];
		m_col[2][i] = value[2];
		m_col[3][i] = value[3];
	}

	constexpr inline auto operator[](this auto&& self, int col) -> auto&& { return std::forward<decltype(self)>(self).m_col[col]; }
	constexpr inline auto operator[](this auto&& self, int row, int col) -> auto&& { return std::forward<decltype(self)>(self).m_col[col][row]; }
	constexpr inline auto operator()(this auto&& self, int row, int col) -> auto&& { return std::forward<decltype(self)>(self).m_col[col][row]; }

	union {
		float4 m_col[4];
		float  M[16];
	};
};

// clang-format off
constexpr inline auto operator==(float2 const& lhs, float2 const& rhs) -> bool { return lhs.x == rhs.x && lhs.y == rhs.y; }
constexpr inline auto operator!=(float2 const& lhs, float2 const& rhs) -> bool { return !(lhs == rhs); }

constexpr inline auto operator+(float2 const& other) -> float2 { return other; }
constexpr inline auto operator-(float2 const& other) -> float2 { return float2(-other.x, -other.y); }

constexpr inline auto operator+=(float2& lhs, float2 const& rhs) -> float2& { lhs.x += rhs.x; lhs.y += rhs.y; return lhs; }
constexpr inline auto operator-=(float2& lhs, float2 const& rhs) -> float2& { lhs.x -= rhs.x; lhs.y -= rhs.y; return lhs; }
constexpr inline auto operator*=(float2& lhs, float2 const& rhs) -> float2& { lhs.x *= rhs.x; lhs.y *= rhs.y; return lhs; }
constexpr inline auto operator/=(float2& lhs, float2 const& rhs) -> float2& { lhs.x /= rhs.x; lhs.y /= rhs.y; return lhs; }
constexpr inline auto operator+=(float2& lhs, float const value) -> float2& { lhs.x += value; lhs.y += value; return lhs; }
constexpr inline auto operator-=(float2& lhs, float const value) -> float2& { lhs.x -= value; lhs.y -= value; return lhs; }
constexpr inline auto operator*=(float2& lhs, float const value) -> float2& { lhs.x *= value; lhs.y *= value; return lhs; }
constexpr inline auto operator/=(float2& lhs, float const value) -> float2& { lhs.x /= value; lhs.y /= value; return lhs; }

constexpr inline auto operator+(float2 const& lhs, float2 const& rhs) -> float2 { return float2(lhs.x + rhs.x, lhs.y + rhs.y); }
constexpr inline auto operator-(float2 const& lhs, float2 const& rhs) -> float2 { return float2(lhs.x - rhs.x, lhs.y - rhs.y); }
constexpr inline auto operator*(float2 const& lhs, float2 const& rhs) -> float2 { return float2(lhs.x * rhs.x, lhs.y * rhs.y); }
constexpr inline auto operator/(float2 const& lhs, float2 const& rhs) -> float2 { return float2(lhs.x / rhs.x, lhs.y / rhs.y); }
constexpr inline auto operator+(float2 const& lhs, float const value) -> float2 { return float2(lhs.x + value, lhs.y + value); }
constexpr inline auto operator-(float2 const& lhs, float const value) -> float2 { return float2(lhs.x - value, lhs.y - value); }
constexpr inline auto operator*(float2 const& lhs, float const value) -> float2 { return float2(lhs.x * value, lhs.y * value); }
constexpr inline auto operator/(float2 const& lhs, float const value) -> float2 { return float2(lhs.x / value, lhs.y / value); }
constexpr inline auto operator+(float const value, float2 const& rhs) -> float2 { return float2(value + rhs.x, value + rhs.y); }
constexpr inline auto operator-(float const value, float2 const& rhs) -> float2 { return float2(value - rhs.x, value - rhs.y); }
constexpr inline auto operator*(float const value, float2 const& rhs) -> float2 { return float2(value * rhs.x, value * rhs.y); }
constexpr inline auto operator/(float const value, float2 const& rhs) -> float2 { return float2(value / rhs.x, value / rhs.y); }
// clang-format on

// clang-format off
constexpr inline auto operator==(float3 const& lhs, float3 const& rhs) -> bool { return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z; }
constexpr inline auto operator!=(float3 const& lhs, float3 const& rhs) -> bool { return lhs.x != rhs.x || lhs.y != rhs.y || lhs.z != rhs.z; }

constexpr inline auto operator+(float3 const& other) -> float3 { return other; }
constexpr inline auto operator-(float3 const& other) -> float3 { return float3(-other.x, -other.y, -other.z); }
constexpr inline auto operator!(float3 const& other) -> float3 { return float3(!other.x, !other.y, !other.z); }

constexpr inline auto operator++(float3& other) -> float3& { ++other.x; ++other.y; ++other.z; return other; }
constexpr inline auto operator--(float3& other) -> float3& { --other.x; --other.y; --other.z; return other; }
constexpr inline auto operator++(float3& other, int) -> float3 { float3 tmp(other); ++other; return tmp; }
constexpr inline auto operator--(float3& other, int) -> float3 { float3 tmp(other); --other; return tmp; }

constexpr inline auto operator+=(float3& lhs, float3 const& rhs) -> float3& { lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; return lhs; }
constexpr inline auto operator-=(float3& lhs, float3 const& rhs) -> float3& { lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; return lhs; }
constexpr inline auto operator*=(float3& lhs, float3 const& rhs) -> float3& { lhs.x *= rhs.x; lhs.y *= rhs.y; lhs.z *= rhs.z; return lhs; }
constexpr inline auto operator/=(float3& lhs, float3 const& rhs) -> float3& { lhs.x /= rhs.x; lhs.y /= rhs.y; lhs.z /= rhs.z; return lhs; }
constexpr inline auto operator+=(float3& lhs, float const value) -> float3& { lhs.x += value; lhs.y += value; lhs.z += value; return lhs; }
constexpr inline auto operator-=(float3& lhs, float const value) -> float3& { lhs.x -= value; lhs.y -= value; lhs.z -= value; return lhs; }
constexpr inline auto operator*=(float3& lhs, float const value) -> float3& { lhs.x *= value; lhs.y *= value; lhs.z *= value; return lhs; }
constexpr inline auto operator/=(float3& lhs, float const value) -> float3& { lhs.x /= value; lhs.y /= value; lhs.z /= value; return lhs; }

constexpr inline auto operator+(float3 const& lhs, float3 const& rhs) -> float3 { return float3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z); }
constexpr inline auto operator-(float3 const& lhs, float3 const& rhs) -> float3 { return float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z); }
constexpr inline auto operator*(float3 const& lhs, float3 const& rhs) -> float3 { return float3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z); }
constexpr inline auto operator/(float3 const& lhs, float3 const& rhs) -> float3 { return float3(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z); }
constexpr inline auto operator+(float3 const& vec, float const value) -> float3 { return float3(vec.x + value, vec.y + value, vec.z + value); }
constexpr inline auto operator-(float3 const& vec, float const value) -> float3 { return float3(vec.x - value, vec.y - value, vec.z - value); }
constexpr inline auto operator*(float3 const& vec, float const value) -> float3 { return float3(vec.x * value, vec.y * value, vec.z * value); }
constexpr inline auto operator/(float3 const& vec, float const value) -> float3 { return float3(vec.x / value, vec.y / value, vec.z / value); }
constexpr inline auto operator+(float const value, float3 const& vec) -> float3 { return float3(value + vec.x, value + vec.y, value + vec.z); }
constexpr inline auto operator-(float const value, float3 const& vec) -> float3 { return float3(value - vec.x, value - vec.y, value - vec.z); }
constexpr inline auto operator*(float const value, float3 const& vec) -> float3 { return float3(value * vec.x, value * vec.y, value * vec.z); }
constexpr inline auto operator/(float const value, float3 const& vec) -> float3 { return float3(value / vec.x, value / vec.y, value / vec.z); }
// clang-format on

// clang-format off
constexpr inline auto operator==(float4 const& lhs, float4 const& rhs) -> bool { return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w; }
constexpr inline auto operator!=(float4 const& lhs, float4 const& rhs) -> bool { return !(lhs == rhs); }

constexpr inline auto operator+(float4 const& other) -> float4 { return other; }
constexpr inline auto operator-(float4 const& other) -> float4 { return float4(-other.x, -other.y, -other.z, -other.w); }

constexpr inline auto operator!(float4 const& other) -> float4 { return float4(!other.x, !other.y, !other.z, !other.w); }
constexpr inline auto operator++(float4& other) -> float4& { ++other.x; ++other.y; ++other.z; ++other.w; return other; }
constexpr inline auto operator--(float4& other) -> float4& { --other.x; --other.y; --other.z; --other.w; return other; }
constexpr inline auto operator++(float4& other, int) -> float4 { float4 tmp(other); ++other; return tmp; }
constexpr inline auto operator--(float4& other, int) -> float4 { float4 tmp(other); --other; return tmp; }

constexpr inline auto operator+=(float4& lhs, float4 const& rhs) -> float4& { lhs.x += rhs.x; lhs.y += rhs.y; lhs.z += rhs.z; lhs.w += rhs.w; return lhs; }
constexpr inline auto operator-=(float4& lhs, float4 const& rhs) -> float4& { lhs.x -= rhs.x; lhs.y -= rhs.y; lhs.z -= rhs.z; lhs.w -= rhs.w; return lhs; }
constexpr inline auto operator*=(float4& lhs, float4 const& rhs) -> float4& { lhs.x *= rhs.x; lhs.y *= rhs.y; lhs.z *= rhs.z; lhs.w *= rhs.w; return lhs; }
constexpr inline auto operator/=(float4& lhs, float4 const& rhs) -> float4& { lhs.x /= rhs.x; lhs.y /= rhs.y; lhs.z /= rhs.z; lhs.w /= rhs.w; return lhs; }
constexpr inline auto operator+=(float4& lhs, float const value) -> float4& { lhs.x += value; lhs.y += value; lhs.z += value; lhs.w += value; return lhs; }
constexpr inline auto operator-=(float4& lhs, float const value) -> float4& { lhs.x -= value; lhs.y -= value; lhs.z -= value; lhs.w -= value; return lhs; }
constexpr inline auto operator*=(float4& lhs, float const value) -> float4& { lhs.x *= value; lhs.y *= value; lhs.z *= value; lhs.w *= value; return lhs; }
constexpr inline auto operator/=(float4& lhs, float const value) -> float4& { lhs.x /= value; lhs.y /= value; lhs.z /= value; lhs.w /= value; return lhs; }

constexpr inline auto operator+(float4 const& lhs, float4 const& rhs) -> float4 { return float4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w); }
constexpr inline auto operator-(float4 const& lhs, float4 const& rhs) -> float4 { return float4(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w); }
constexpr inline auto operator*(float4 const& lhs, float4 const& rhs) -> float4 { return float4(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w); }
constexpr inline auto operator/(float4 const& lhs, float4 const& rhs) -> float4 { return float4(lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w); }
constexpr inline auto operator+(float4 const& vec, float const value) -> float4 { return float4(vec.x + value, vec.y + value, vec.z + value, vec.w + value); }
constexpr inline auto operator-(float4 const& vec, float const value) -> float4 { return float4(vec.x - value, vec.y - value, vec.z - value, vec.w - value); }
constexpr inline auto operator*(float4 const& vec, float const value) -> float4 { return float4(vec.x * value, vec.y * value, vec.z * value, vec.w * value); }
constexpr inline auto operator/(float4 const& vec, float const value) -> float4 { return float4(vec.x / value, vec.y / value, vec.z / value, vec.w / value); }
constexpr inline auto operator+(float const value, float4 const& vec) -> float4 { return float4(value + vec.x, value + vec.y, value + vec.z, value + vec.w); }
constexpr inline auto operator-(float const value, float4 const& vec) -> float4 { return float4(value - vec.x, value - vec.y, value - vec.z, value - vec.w); }
constexpr inline auto operator*(float const value, float4 const& vec) -> float4 { return float4(value * vec.x, value * vec.y, value * vec.z, value * vec.w); }
constexpr inline auto operator/(float const value, float4 const& vec) -> float4 { return float4(value / vec.x, value / vec.y, value / vec.z, value / vec.w); }

// clang-format on

inline float3x3::float3x3(float4x4 const& from4x4) {
	m_col[0] = from4x4.m_col[0].xyz;
	m_col[1] = from4x4.m_col[1].xyz;
	m_col[2] = from4x4.m_col[2].xyz;
}

constexpr inline float  dot(float3 const& a, float3 const& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
constexpr inline float3 abs(float3 const& a) { return float3{std::abs(a.x), std::abs(a.y), std::abs(a.z)}; }
constexpr inline float  length(float3 const& a) { return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }
constexpr inline float3 normalize(float3 const& a) { return a / length(a); }
constexpr inline float3 clamp(float3 const& a, float3 const& min, float3 const& max) { return float3{std::clamp(a.x, min.x, max.x), std::clamp(a.y, min.y, max.y), std::clamp(a.z, min.z, max.z)}; }
constexpr inline float3 lerp(float3 const& a, float3 const& b, float const t) { return float3{std::lerp(a.x, b.x, t), std::lerp(a.y, b.y, t), std::lerp(a.z, b.z, t)}; }

inline float3 shuffle_xzy(float3 a) { return float3{a.x, a.z, a.y}; }
inline float3 shuffle_yxz(float3 a) { return float3{a.y, a.x, a.z}; }
inline float3 shuffle_yzx(float3 a) { return float3{a.y, a.z, a.x}; }
inline float3 shuffle_zxy(float3 a) { return float3{a.z, a.x, a.y}; }
inline float3 shuffle_zyx(float3 a) { return float3{a.z, a.y, a.x}; }

inline float3 cross(float3 const a, float3 const b) {
	float3 const a_yzx = shuffle_yzx(a);
	float3 const b_yzx = shuffle_yzx(b);
	return shuffle_yzx(a * b_yzx - a_yzx * b);
}

// RH
inline float4x4 lookAt(float3 eye, float3 center, float3 up) {
	float3 r, u, f;

	f = normalize(center - eye);
	r = normalize(cross(f, up));
	u = normalize(cross(r, f));

	return float4x4{
		{r.x, u.x, f.x, 0.0f},
		{r.y, u.y, f.y, 0.0f},
		{r.z, u.z, f.z, 0.0f},
		{-dot(r, eye), -dot(u, eye), -dot(f, eye), 1.0f},
	};
}

inline float4x4 lookAtInverse(float3 eye, float3 center, float3 up) {
	float3 const f = normalize(center - eye);
	float3 const s = normalize(cross(f, up));
	float3 const u = cross(s, f);
	float4x4     res;
	res.m_col[0] = float4{s.x, s.y, s.z, dot(s, eye)};
	res.m_col[1] = float4{u.x, u.y, u.z, dot(u, eye)};
	res.m_col[2] = float4{f.x, f.y, f.z, dot(f, eye)};
	res.m_col[3] = float4{0.0f, 0.0f, 0.0f, 1.0f};
	return res;
}

inline float4x4 perspectiveX(float fov_x, float aspect, float z_near, float z_far) {
	float tan_half_fov_inverse = 1.0f / std::tanf(fov_x * DEG_TO_RAD / 2.0f);

	float4x4 result;
	result[0, 0] = tan_half_fov_inverse;
	result[1, 1] = aspect * tan_half_fov_inverse;
	result[2, 2] = -(z_far + z_near) / (z_far - z_near);
	result[3, 2] = -1.0f;
	result[2, 3] = -(2.0f * z_far * z_near) / (z_far - z_near);
	result[3, 3] = 0.0f;
	return result;
}

inline float4x4 perspectiveY(float fov_y, float aspect, float z_near, float z_far) {
	float tan_half_fov_inverse = 1.0f / std::tanf(fov_y * DEG_TO_RAD / 2.0f);

	float4x4 result;
	result[0, 0] = aspect * tan_half_fov_inverse;
	result[1, 1] = tan_half_fov_inverse;
	result[2, 2] = -(z_far + z_near) / (z_far - z_near);
	result[3, 2] = -1.0f;
	result[2, 3] = -(2.0f * z_far * z_near) / (z_far - z_near);
	result[3, 3] = 0.0f;
	return result;
}
inline float3x3 transpose(float3x3 const& m1) {
	float3x3 res;
	res[0] = float3{m1[0, 0], m1[1, 0], m1[2, 0]};
	res[1] = float3{m1[0, 1], m1[1, 1], m1[2, 1]};
	res[2] = float3{m1[0, 2], m1[1, 2], m1[2, 2]};
	return res;
}

inline float3x3 inverse(float3x3 const& m) {
	float a = m.m_col[0].x, b = m.m_col[0].y, c = m.m_col[0].z;
	float d = m.m_col[1].x, e = m.m_col[1].y, f = m.m_col[1].z;
	float g = m.m_col[2].x, h = m.m_col[2].y, i = m.m_col[2].z;

	float det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);

	float inv_det = 1.0f / det;

	float3x3 res;

	res.m_col[0].x = (e * i - f * h) * inv_det;
	res.m_col[0].y = (c * h - b * i) * inv_det;
	res.m_col[0].z = (b * f - c * e) * inv_det;

	res.m_col[1].x = (f * g - d * i) * inv_det;
	res.m_col[1].y = (a * i - c * g) * inv_det;
	res.m_col[1].z = (c * d - a * f) * inv_det;

	res.m_col[2].x = (d * h - e * g) * inv_det;
	res.m_col[2].y = (b * g - a * h) * inv_det;
	res.m_col[2].z = (a * e - b * d) * inv_det;

	return res;
}

inline void mat3_colmajor_mul_vec3(float* __restrict RES, const float* __restrict B, const float* __restrict V) {
	RES[0] = V[0] * B[0] + V[1] * B[3] + V[2] * B[6];
	RES[1] = V[0] * B[1] + V[1] * B[4] + V[2] * B[7];
	RES[2] = V[0] * B[2] + V[1] * B[5] + V[2] * B[8];
}

inline float3 mul(float3x3 const& m, float3 const& v) {
	float3 res;
	mat3_colmajor_mul_vec3((float*)&res, (const float*)&m, (const float*)&v);
	return res;
}

inline float3 operator*(float3x3 const& m, float3 const& v) { return mul(m, v); }

inline void mat4_colmajor_mul_vec4(float* __restrict res, const float* __restrict b, const float* __restrict v) {
	res[0] = v[0] * b[0] + v[1] * b[4] + v[2] * b[8] + v[3] * b[12];
	res[1] = v[0] * b[1] + v[1] * b[5] + v[2] * b[9] + v[3] * b[13];
	res[2] = v[0] * b[2] + v[1] * b[6] + v[2] * b[10] + v[3] * b[14];
	res[3] = v[0] * b[3] + v[1] * b[7] + v[2] * b[11] + v[3] * b[15];
}

inline float4 mul(float4x4 const& m, float4 const& v) {
	float4 res;
	mat4_colmajor_mul_vec4((float*)&res, (const float*)&m, (const float*)&v);
	return res;
}

inline float4x4 translate4x4(float3 t) {
	float4x4 res;
	res[3] = float4{t.x, t.y, t.z, 1.0f};
	return res;
}

inline float4x4 scale4x4(float3 t) {
	float4x4 res;
	res[0] = float4{t.x, 0.0f, 0.0f, 0.0f};
	res[1] = float4{0.0f, t.y, 0.0f, 0.0f};
	res[2] = float4{0.0f, 0.0f, t.z, 0.0f};
	res[3] = float4{0.0f, 0.0f, 0.0f, 1.0f};
	return res;
}

inline float4x4 rotate4x4X(float phi) {
	float4x4 res;
	res[0] = float4{1.0f, 0.0f, 0.0f, 0.0f};
	res[1] = float4{0.0f, +cosf(phi), +sinf(phi), 0.0f};
	res[2] = float4{0.0f, -sinf(phi), +cosf(phi), 0.0f};
	res[3] = float4{0.0f, 0.0f, 0.0f, 1.0f};
	return res;
}

inline float4x4 rotate4x4Y(float phi) {
	float4x4 res;
	res[0] = float4{+cosf(phi), 0.0f, -sinf(phi), 0.0f};
	res[1] = float4{0.0f, 1.0f, 0.0f, 0.0f};
	res[2] = float4{+sinf(phi), 0.0f, +cosf(phi), 0.0f};
	res[3] = float4{0.0f, 0.0f, 0.0f, 1.0f};
	return res;
}

inline float4x4 rotate4x4Z(float phi) {
	float4x4 res;
	res[0] = float4{+cosf(phi), sinf(phi), 0.0f, 0.0f};
	res[1] = float4{-sinf(phi), cosf(phi), 0.0f, 0.0f};
	res[2] = float4{0.0f, 0.0f, 1.0f, 0.0f};
	res[3] = float4{0.0f, 0.0f, 0.0f, 1.0f};
	return res;
}

// Rotation around an arbitrary axis
inline float4x4 rotate4x4(float3 axis, float angle) {
	axis      = normalize(axis);
	float c   = cosf(angle);
	float s   = sinf(angle);
	float omc = 1.0f - c;

	return float4x4{
		{axis.x * axis.x * omc + c, axis.x * axis.y * omc - axis.z * s, axis.x * axis.z * omc + axis.y * s, 0.0f},
		{axis.x * axis.y * omc + axis.z * s, axis.y * axis.y * omc + c, axis.y * axis.z * omc - axis.x * s, 0.0f},
		{axis.x * axis.z * omc - axis.y * s, axis.y * axis.z * omc + axis.x * s, axis.z * axis.z * omc + c, 0.0f},
		{0.0f, 0.0f, 0.0f, 1.0f}};
}

inline float4x4 mul(float4x4 m1, float4x4 m2) {
	float4 const column1 = mul(m1, m2.col(0));
	float4 const column2 = mul(m1, m2.col(1));
	float4 const column3 = mul(m1, m2.col(2));
	float4 const column4 = mul(m1, m2.col(3));
	float4x4     res;
	res[0] = column1;
	res[1] = column2;
	res[2] = column3;
	res[3] = column4;

	return res;
}

inline float4x4 operator*(float4x4 m1, float4x4 m2) {
	float4 const column1 = mul(m1, m2.col(0));
	float4 const column2 = mul(m1, m2.col(1));
	float4 const column3 = mul(m1, m2.col(2));
	float4 const column4 = mul(m1, m2.col(3));

	return float4x4{column1, column2, column3, column4};
}

inline float4x4 make_float4x4(const float* p) {
	float4x4 r;
	std::memcpy((void*)&r, (void*)p, sizeof(r));
	return r;
}

inline float4x4 transpose(float4x4 const& m1) {
	float4x4 res;
	res.set_row(0, float4{m1(0, 0), m1(1, 0), m1(2, 0), m1(3, 0)});
	res.set_row(1, float4{m1(0, 1), m1(1, 1), m1(2, 1), m1(3, 1)});
	res.set_row(2, float4{m1(0, 2), m1(1, 2), m1(2, 2), m1(3, 2)});
	res.set_row(3, float4{m1(0, 3), m1(1, 3), m1(2, 3), m1(3, 3)});
	return res;
}

inline float4x4 affineInverse(float4x4 const& m) {
	auto inv = inverse(float3x3(m));
	auto l   = -(inv * m.m_col[3].xyz);

	return float4x4{
		float4{inv.m_col[0], 0.0f},
		float4{inv.m_col[1], 0.0f},
		float4{inv.m_col[2], 0.0f},
		float4{l.x, l.y, l.z, 1.0f},
	};
}

inline float4x4 operator|(float4x4 const& m, float4x4 (*func)(float4x4 const&)) {
	return func(m);
}
struct translate_op {
	float3 translation;
	inline translate_op(float3 t) : translation(t) {}
};

inline translate_op translate(float3 t) { return translate_op(t); }
inline float4x4     operator|(float4x4 m, translate_op const& op) {
    m.col(3) += float4{op.translation.x, op.translation.y, op.translation.z, 0.0f};
    return m;
}

struct rotateX_op {
	float rotation;
};
inline rotateX_op rotateX(float x) { return {x}; }
inline float4x4   operator|(float4x4 m, rotateX_op const& op) { return rotate4x4X(op.rotation) * m; }
inline float4x4&  operator|=(float4x4& m, rotateX_op const& op) { return m = rotate4x4X(op.rotation) * m; }

struct rotateY_op {
	float rotation;
};
inline rotateY_op rotateY(float x) { return {x}; }
inline float4x4   operator|(float4x4 m, rotateY_op const& op) { return rotate4x4Y(op.rotation) * m; }
inline float4x4&  operator|=(float4x4& m, rotateY_op const& op) { return m = rotate4x4Y(op.rotation) * m; }

struct rotateZ_op {
	float rotation;
};
inline rotateZ_op rotateZ(float x) { return {x}; }
inline float4x4   operator|(float4x4 m, rotateZ_op const& op) { return rotate4x4Z(op.rotation) * m; }
inline float4x4&  operator|=(float4x4& m, rotateZ_op const& op) { return m = rotate4x4Z(op.rotation) * m; }

struct rotate_euler_op {
	float3 axis;
	float  rotation;
};
inline rotate_euler_op rotate(float3 a, float r) { return {a, r}; }
inline float4x4        operator|(float4x4 const& m, rotate_euler_op const& op) { return rotate4x4(op.axis, op.rotation) * m; }

} // namespace math
