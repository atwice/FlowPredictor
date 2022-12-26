#pragma once

class Model {
public:
	// В конструкторе загружаем модель из директории (формат Tensorflow::SAVED_MODEL)
	Model(const std::filesystem::path& modelPath, const std::filesystem::path& logPath);

	void SetInputLayer(const std::string& _inputLayer) { inputLayer = _inputLayer; }

	// Получить предсказание по модели
	double Predict(int vectorSize, const double* vector);
	bool Predict(int shapeSize, const int* shape, const double* tensor,
		int outVectorSize, float* outVector);

private:
	cppflow::model tensorflowModel;
	std::wofstream logStream;
	std::string inputLayer;

	void logNodeNames();
	void logPredict(const std::vector<int64_t>& shape, const std::vector<double>& data);
	void logResult(const std::vector<float>&);
};
