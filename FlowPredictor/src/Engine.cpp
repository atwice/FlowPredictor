#include <pch.h>
#include <Engine.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace fs = std::filesystem;

extern HMODULE CurrentDllModule;

static std::filesystem::path getDllPath()
{
	wchar_t rawDllPath[512 + 1] = { 0 };
	GetModuleFileName(CurrentDllModule, rawDllPath, 512);
	std::filesystem::path dllFilePath(rawDllPath);
	return dllFilePath.parent_path();
}

// ------------------------------------------------------------------------------------------------

Engine::Engine() :
	dllPath( getDllPath() )
{
}

void Engine::SetLogPath(const wchar_t* path)
{
	if (path == 0 || *path == 0) {
		logsPath = std::filesystem::path();
		return;
	}
	wchar_t expandedPath[512 + 1] = { 0 };
	ExpandEnvironmentStringsW(path, expandedPath, 512);

	logsPath = expandedPath;

	if (logsPath.is_relative()) {
		// Если задан относительный путь, то считаем относительно расположения DLL
		logsPath = dllPath / logsPath;
	}
}


void Engine::SetModelPath(const wchar_t* path)
{
	wchar_t expandedPath[512 + 1] = { 0 };
	ExpandEnvironmentStringsW(path, expandedPath, 512);

	modelsPath = expandedPath;

	if (modelsPath.is_relative()) {
		// Если задан относительный путь, то считаем относительно расположения DLL
		modelsPath = dllPath / modelsPath;
	}
}

static inline bool isFileExist(const fs::path& filePath)
{
	fs::file_status fileStatus = fs::status(filePath);
	return fileStatus.type() != fs::file_type::not_found;
}

bool Engine::Load(const wchar_t* _modelName)
{
	std::wstring modelName = _modelName;
	if (models.count(modelName) > 0) {
		return true;
	}
	fs::path modelPath = modelsPath / modelName;
	fs::path logPath;
	std::wofstream logStream;
	
	if (!logsPath.empty()) {
		logPath = logsPath / modelName;
		logPath.replace_extension(".log.txt");
		logStream.rdbuf()->open(logPath, std::ios::app);
		logStream << L"Model: " << _modelName << std::endl;
	}
	
	if (!isFileExist(modelPath)) {
		logStream << L"File not found: " << modelPath << std::endl;
		return false;
	}

	std::unique_ptr<Model> model;
	try {
		model = std::make_unique<Model>(modelPath, logPath);
	} catch (std::runtime_error& err) {
		logStream << L"Cant load model:" << std::endl;
		logStream << err.what() << std::endl;
		return false;
	}
	
	models[modelName] = std::move(model);
	return true;
}

void Engine::Release(const wchar_t* modelName)
{
	models.erase(modelName);
}

double Engine::Predict(const wchar_t* modelName, int vectorSize, const double* vector)
{
	return models[modelName]->Predict(vectorSize, vector);
}

