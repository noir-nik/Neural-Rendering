 
export module UI;

export
namespace UI {

struct Context {
	void Init(void* imguiStyle = nullptr);
	void Destroy();
	void SetCurrent(); // Thanks ImGui

	void* GetHandle() { return context; }
private:
	void* context = nullptr;
};


void Init();
void Destroy();

bool SaveStyle();
bool LoadStyle();
} // namespace UI