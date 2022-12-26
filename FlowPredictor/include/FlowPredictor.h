#pragma once

#ifdef FLOWPREDICTOR_EXPORTS
#define PREDICTOR_API __declspec(dllexport)
#else
#define PREDICTOR_API __declspec(dllimport)
#endif // FLOWPREDICTOR_EXPORTS

extern "C" {
	// Установить путь к директории с отлажочными журналами. Можно относительный - тогда считаем относительно расположения DLL
	PREDICTOR_API void __stdcall FlowPredictor_SetLogPath(const wchar_t* path);
	
	// Установить путь к директории с моделями. Можно относительный - тогда считаем относительно расположения DLL
	PREDICTOR_API void __stdcall FlowPredictor_SetModelPath(const wchar_t* path);

	// Загрузить модель. Формат TensorFlow::SAVED_MODEL
	// Имя модели совпадает с именем директории, в которой расположен файл saved_model.pb
	PREDICTOR_API bool __stdcall FlowPredictor_LoadModel(const wchar_t* modelName);

	// То же, что предыдущий метод, но позволяет установить имя входного слоя
	PREDICTOR_API bool __stdcall FlowPredictor_LoadModelWithInput(
		const wchar_t* modelName, const wchar_t* inputLayer);

	// Удалить ранее загруженную модель
	PREDICTOR_API void __stdcall FlowPredictor_ReleaseModel(const wchar_t* modelName);

	// Получить предсказание модели по вектору признаков
	PREDICTOR_API double __stdcall FlowPredictor_Predict(const wchar_t* modelName,
		int vectorSize, const double* vector);

	// Получить предсказание модели по тензору
	// Входные параметры:
	// `shapeSize` и `shape` описывают размерность входного тензора
	// `shapeSize` - количество измерений тензора, то есть размер входного массива `shape`
	// `shape` - размеры тензора по разным измерениям. Например, {1024, 5, 3}
	// `tensor` - сами признаким
	// Результат
	// `outVector` - массив, в который будет записан результат
	// `outVectorSize` - размер выходного массива `outVector`, в который записан резльтат
	//
	// Возвращает `true`, если всё хорошо.
	// `false` - если размер входного тензора не совпадает с ожиданиями модели,
	// или размер `outVectorSize` не совпадает с размером выхода модели
	PREDICTOR_API bool __stdcall FlowPredictor_PredictTensor(const wchar_t* modelName,
		int shapeSize, const int* shape, const double* tensor,
		int outVectorSize, float* outVector);
}
