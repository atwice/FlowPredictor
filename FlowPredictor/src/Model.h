#pragma once

class Model {
public:
	// В конструкторе загружаем модель из директории (формат Tensorflow::SAVED_MODEL)
	Model(const std::filesystem::path& modelPath, const std::filesystem::path& logPath);

	// Получить предсказание по модели
	double Predict(int vectorSize, const double* vector);

private:
	cppflow::model tensorflowModel;
	std::wofstream logStream;

	void logNodeNames();
	void logPredict(int vectorSize, const double* vector);
};
