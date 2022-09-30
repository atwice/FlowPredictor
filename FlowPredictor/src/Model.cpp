#include <pch.h>
#include <Model.h>

const char* const InputNodeName = "serving_default_input_input";
const char* const OutputNodeName = "StatefulPartitionedCall";

Model::Model(const std::filesystem::path& path) :
	tensorflowModel(path.string())
{
}

double Model::Predict(int vectorSize, const double* vector)
{
	std::vector<double> inputVector(vector, vector + vectorSize);
	std::vector<int64_t> shape({ vectorSize });
	cppflow::tensor input(inputVector, shape);
	auto output = tensorflowModel({ std::make_tuple(InputNodeName, input) }, { OutputNodeName });
	std::vector<float> result = output[0].get_data<float>();
	return result[0];
}
