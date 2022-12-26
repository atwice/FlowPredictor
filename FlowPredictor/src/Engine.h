#pragma once
#include <Model.h>

class Engine {
public:
	// Синглтон
	static Engine& Instance()
	{
		static Engine instance;
		return instance;
	}

	void SetLogPath(const wchar_t* path);
	// установить путь к директории с моделями
	void SetModelPath(const wchar_t* path);
	// Загрузить модель
	bool Load(const wchar_t* modelName, const wchar_t* inputLayer = nullptr);
	// Удалить модель
	void Release(const wchar_t* modelName);
	// Сделать предсказание
	double Predict(const wchar_t* modelName, int vectorSize, const double* vector);
	bool Predict(const wchar_t* modelName,
		int shapeSize, const int* shape, const double* tensor,
		int outVectorSize, float* outVector);

private:
	// директория, в которой хранится dll
	const std::filesystem::path dllPath;
	// директория, в которую пишем логи
	std::filesystem::path logsPath;
	// директория, в которой ищем модели
	std::filesystem::path modelsPath;
	// словарь с загруженными моделями
	std::unordered_map<std::wstring, std::unique_ptr<Model>> models;

	Engine();
};