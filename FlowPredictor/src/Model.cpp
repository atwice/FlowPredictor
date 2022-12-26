#include <pch.h>
#include <Model.h>
#include <functional>
#include <numeric>

const char* const InputNodeName = "serving_default_input_input";
const char* const OutputNodeName = "StatefulPartitionedCall";

Model::Model(const std::filesystem::path& _modelPath, const std::filesystem::path& _logPath) :
	tensorflowModel(_modelPath.string()),
	logStream(_logPath, std::ios::app),
	inputLayer(InputNodeName)
{
	logNodeNames();
}

double Model::Predict(int vectorSize, const double* vector)
{
	std::vector<double> inputVector( vector, vector + vectorSize );
	std::vector<int64_t> shape( { vectorSize } );
	cppflow::tensor input( inputVector, shape );

	logPredict(shape, inputVector);

	if (vectorSize == 0) {
		return 0.0;
	}

	try {
		auto output = tensorflowModel({ std::make_tuple(inputLayer, input) }, { OutputNodeName });
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

bool Model::Predict(int shapeSize, const int* shape, const double* tensor,
		int outVectorSize, float* outVector)
{
	std::vector<int64_t> shapeVec(shape, shape + shapeSize);
	int64_t totalElements = std::accumulate(shapeVec.begin(), shapeVec.end(), 1i64, std::multiplies<int64_t>());
	std::vector<double> inputTensor(tensor, tensor + totalElements);

	cppflow::tensor input(inputTensor, shapeVec);
	try {
		auto output = tensorflowModel( { std::make_tuple(inputLayer, input) }, { OutputNodeName } );
		std::vector<float> resultVector = output[0].get_data<float>();
		logResult(resultVector);
		
		if( resultVector.size() != outVectorSize ) {
			logStream << L"Wrong output vector size:" << outVectorSize << std::endl;
			logStream << L"Actual result size:" << resultVector.size() << std::endl;
			return false;
		}
		std::copy(resultVector.begin(), resultVector.end(), outVector);

	} catch( std::runtime_error& err ) {
		logStream << L"Cant execute model:" << std::endl;
		logStream << err.what() << std::endl;
		return false;
	}
	return true;
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

void Model::logPredict(const std::vector<int64_t>& shape, const std::vector<double>& data)
{
	if(!logStream.is_open()) {
		return;
	}
	std::ostream_iterator<float, wchar_t>(logStream, L", ");
	logStream << L"Predict vector[";
	std::copy(shape.begin(), shape.end(), std::ostream_iterator<int64_t, wchar_t>(logStream, L", "));
	logStream << "] = {" << std::endl << L'\t';
	auto first10 = data.begin() + std::min<std::size_t>({ 10, data.size() });
	std::copy(data.begin(), first10, std::ostream_iterator<double, wchar_t>(logStream, L", "));
	if(data.size() > 10) {
		logStream << L", ...";
	}
	logStream << std::endl << "}";
}

void Model::logResult(const std::vector<float>& result)
{
	if(!logStream.is_open()) {
		return;
	}
	logStream << L" -> [ ";
	std::copy(result.begin(), result.end(),
		std::ostream_iterator<float, wchar_t>(logStream, L", "));
	logStream << L" ]";
}
