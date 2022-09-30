#include <pch.h>
#include <FlowPredictor.h>
#include <Engine.h>

void FlowPredictor_SetModelPath(const wchar_t* path)
{
	Engine::Instance().SetModelPath(path);
}

void FlowPredictor_LoadModel(const wchar_t* modelName)
{
	Engine::Instance().Load(modelName);
}

void FlowPredictor_ReleaseModel(const wchar_t* modelName)
{
	Engine::Instance().Release(modelName);
}

double FlowPredictor_Predict(const wchar_t* modelName, int vectorSize, const double* vector)
{
	return Engine::Instance().Predict(modelName, vectorSize, vector);
}
