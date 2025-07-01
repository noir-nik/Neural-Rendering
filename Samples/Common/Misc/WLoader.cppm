export module WeightsLoader;
import std;

export class WeightsLoader {
	using u32 = std::uint32_t;
public:
	// WeightsLoader(std::string_view filename) { Init(filename); }
	[[nodiscard]]
	bool Init(std::string_view filename, std::string_view header = "");

	u32 NextRows() const { return rows; } // returns 0 on error
	u32 NextCols() const { return cols; } // returns 0 on error
	bool          LoadNext(float* weights, float* bias);
	bool          HasNext() const { return layers > 0; }

	auto GetFileSize() const -> std::size_t { return file_size; }

private:
	std::ifstream file;
	u32 rows      = 0;
	u32 cols      = 0;
	u32 layers    = 0;
	std::size_t   file_size = 0;

	void ReadNextLayerInfo();
};
