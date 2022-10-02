#include <pch.h>
#include <FlowPredictor.h>
#include <Engine.h>

extern "C" void __stdcall FlowPredictor_SetModelPath(const wchar_t* path)
{
	Engine::Instance().SetModelPath(path);
}

extern "C" void __stdcall FlowPredictor_LoadModel(const wchar_t* modelName)
{
	Engine::Instance().Load(modelName);
}

extern "C" void __stdcall FlowPredictor_ReleaseModel(const wchar_t* modelName)
{
	Engine::Instance().Release(modelName);
}

extern "C" double __stdcall FlowPredictor_Predict(const wchar_t* modelName, int vectorSize, const double* vector)
{
	return Engine::Instance().Predict(modelName, vectorSize, vector);
}
