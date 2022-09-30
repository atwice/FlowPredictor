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

	// установить путь к директории с моделями
	void SetModelPath(const wchar_t* path);
	// Загрузить модель
	void Load(const wchar_t* modelName);
	// Удалить модель
	void Release(const wchar_t* modelName);
	// Сделать предсказание
	double Predict(const wchar_t* modelName, int vectorSize, const double* vector);

private:
	// директория, в которой ищем модели
	std::filesystem::path modelsPath;
	// словарь с загруженными моделями
	std::unordered_map<std::wstring, std::unique_ptr<Model>> models;

	Engine() = default;
};