#include <pch.h>
#include <Engine.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

extern HMODULE CurrentDllModule;

void Engine::SetModelPath(const wchar_t* path)
{
	modelsPath = path;

	if (modelsPath.is_relative()) {
		// Если задан относительный путь, то считаем относительно расположения DLL
		wchar_t rawDllPath[512+1] = {0};
		GetModuleFileName(CurrentDllModule, rawDllPath, 512);
		std::filesystem::path dllPath( rawDllPath );
		modelsPath = dllPath.parent_path() / modelsPath;
	}
}

void Engine::Load(const wchar_t* _modelName)
{
	std::wstring modelName = _modelName;
	if (models.count(modelName) > 0) {
		return;
	}
	std::filesystem::path path = modelsPath / modelName;
	models[modelName] = std::make_unique<Model>(path.wstring());
}

void Engine::Release(const wchar_t* modelName)
{
	models.erase(modelName);
}

double Engine::Predict(const wchar_t* modelName, int vectorSize, const double* vector)
{
	return models[modelName]->Predict(vectorSize, vector);
}