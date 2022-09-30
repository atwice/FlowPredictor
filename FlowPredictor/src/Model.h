#pragma once

class Model {
public:
	// В конструкторе загружаем модель из директории (формат Tensorflow::SAVED_MODEL)
	Model(const std::filesystem::path&);
	// Получить предсказание по модели
	double Predict(int vectorSize, const double* vector);

private:
	cppflow::model tensorflowModel;
};
