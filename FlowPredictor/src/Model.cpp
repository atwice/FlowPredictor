#include <pch.h>
#include <Model.h>

const char* const InputNodeName = "serving_default_input_input";
const char* const OutputNodeName = "StatefulPartitionedCall";

Model::Model(const std::filesystem::path& _modelPath, const std::filesystem::path& _logPath) :
	tensorflowModel(_modelPath.string()),
	logStream(_logPath, std::ios::app)
{
	logNodeNames();
}

double Model::Predict(int vectorSize, const double* vector)
{
	logPredict(vectorSize, vector);

	if (vectorSize == 0) {
		return 0.0;
	}

	try {
		std::vector<double> inputVector(vector, vector + vectorSize);
		std::vector<int64_t> shape({ vectorSize });
		cppflow::tensor input(inputVector, shape);
		auto output = tensorflowModel({ std::make_tuple(InputNodeName, input) }, { OutputNodeName });
		std::vector<float> resultVector = output[0].get_data<float>();
		const double result = resultVector[0];
		logStream << " -> " << result << std::endl;
		return result;

	} catch (std::runtime_error& err) {
		logStream << L"Cant execute model:" << std::endl;
		logStream << err.what() << std::endl;
		return 0.0;
	}
}

void Model::logNodeNames()
{
	if (!logStream.is_open()) {
		return;
	}
	logStream << L"----Operations:-----" << std::endl;
	for (const std::string& op : tensorflowModel.get_operations()) {
		if (op == "NoOp") {
			continue; // NoOp doesn't have a shape
		}
		logStream << L"\t" << op.c_str() << L" shape[ ";
		for (int64_t dimensionSize : tensorflowModel.get_operation_shape(op)) {
			logStream << dimensionSize << " ";
		}
		logStream << "]" << std::endl;
	}
	logStream << L"---------------------" << std::endl;
	logStream << std::endl;
}

void Model::logPredict(int vectorSize, const double* vector)
{
	if (!logStream.is_open()) {
		return;
	}
	logStream << L"Predict vector[" << vectorSize << "] = {" << std::endl << L'\t';
	for (int i = 0; i < vectorSize; i++) {
		if (i > 0) {
			logStream << ", ";
		}
		logStream << vector[i];
	}
	logStream << std::endl << "}";
}
