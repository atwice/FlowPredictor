#include <pch.h>
#include <FlowPredictor.h>
#include <Engine.h>

extern "C" void __stdcall FlowPredictor_SetLogPath(const wchar_t* path)
{
	Engine::Instance().SetLogPath(path);
}

extern "C" void __stdcall FlowPredictor_SetModelPath(const wchar_t* path)
{
	Engine::Instance().SetModelPath(path);
}

extern "C" bool __stdcall FlowPredictor_LoadModel(const wchar_t* modelName)
{
	return Engine::Instance().Load(modelName);
}

extern "C" bool __stdcall FlowPredictor_LoadModelWithInput(
	const wchar_t* modelName, const wchar_t* inputLayer)
{
	return Engine::Instance().Load(modelName, inputLayer);
}

extern "C" void __stdcall FlowPredictor_ReleaseModel(const wchar_t* modelName)
{
	Engine::Instance().Release(modelName);
}

extern "C" double __stdcall FlowPredictor_Predict(const wchar_t* modelName, int vectorSize, const double* vector)
{
	return Engine::Instance().Predict(modelName, vectorSize, vector);
}

extern "C" bool  __stdcall FlowPredictor_PredictTensor(
		const wchar_t* modelName,
		int shapeSize, const int* shape, const double* tensor,
		int outVectorSize, float* outVector)
{
	return Engine::Instance().Predict(modelName,
		shapeSize, shape, tensor,
		outVectorSize, outVector);
}
