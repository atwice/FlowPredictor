#pragma once

#ifdef FLOWPREDICTOR_EXPORTS
#define PREDICTOR_API __declspec(dllexport)
#else
#define PREDICTOR_API __declspec(dllimport)
#endif // FLOWPREDICTOR_EXPORTS

// Установить путь к директории с моделями. Можно относительный - тогда считаем относительно расположения DLL
PREDICTOR_API void FlowPredictor_SetModelPath(const wchar_t* path);

// Загрузить модель. Формат TensorFlow::SAVED_MODEL
// Имя модели совпадает с именем директории, в которой расположен файл saved_model.pb
PREDICTOR_API void FlowPredictor_LoadModel(const wchar_t* modelName);

// Удалить ранее загруженную модель
PREDICTOR_API void FlowPredictor_ReleaseModel(const wchar_t* modelName);

// Получить предсказание модели по вектору признаков
PREDICTOR_API double FlowPredictor_Predict(const wchar_t* modelName, int vectorSize, const double* vector);
