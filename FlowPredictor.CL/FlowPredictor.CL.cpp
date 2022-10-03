#include <iostream>
#include <vector>
#include <FlowPredictor.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <sstream>

const wchar_t* const ModelName = L"BaseCross-ds-m1-m30-v01";

static inline double convertToDouble(const std::string& s)
{
    std::istringstream i(s);
    double x;
    i >> x;
    return x;
}

void processCsv(std::filesystem::path path)
{
    std::ifstream inputStream(path.string());
    std::ofstream outputStream(path.string() + ".result.csv", std::ios::out);
    
    std::string line;
    std::vector<double> features;
    
    // пропустить строку заголовка
    std::getline(inputStream, line);
    outputStream << line << ", Result" << std::endl;

    while (!inputStream.eof()) {
        features.clear();

        std::getline(inputStream, line);
        std::istringstream iss(line);
        std::string item;
        while (std::getline(iss, item, ',')) {
            features.push_back(convertToDouble(item));
        }
        if (features.size() == 0) {
            continue;
        }

        double prediction = FlowPredictor_Predict(ModelName, static_cast<int>(features.size()), features.data());
        outputStream << line << ", " << prediction << std::endl;;
    }
}

void processOneVector()
{
    std::vector<double> test{
        //-0.00036,0.00013,388,-0.00150,0.00026,0.00001,2,52,52,2,0,0.00000,-0.00036,0.00013,0,0,0,0,24,0.00021,-0.00097,0.00044,0,0,9,0,272,-0.00103,-0.00091,0.00012,0,136,80,24,269,-0.00095,-0.00100,0.00015,0,135,80,45,126,-0.00151,0.00091,0.00039,36,0,0,12,534,-0.00009,-0.00159,0.00086,70,120,20,30,732,-0.00205,0.00283,0.00077,0,48,36,24,414,-0.00204,0.00247,0.00144,0,0,60,30,584,-0.00423,0.00577,0.00167,0,0,0,40,384,-0.00025,-0.00152,0.00040,120,300,270,60
        //-0.00075,0.00011,394,-0.00163,0.00026,0.00001,2,52,52,2,6,-0.00013,-0.00036,0.00013,4,0,0,4,30,0.00008,-0.00097,0.00044,0,0,9,33,278,-0.00116,-0.00091,0.00012,0,136,80,24,0,0.00000,-0.00075,0.00011,0,0,0,0,132,-0.00164,0.00091,0.00039,36,0,0,12,540,-0.00022,-0.00159,0.00086,70,120,20,30,738,-0.00218,0.00283,0.00077,0,48,36,24,420,-0.00216,0.00247,0.00144,0,0,60,30,590,-0.00436,0.00577,0.00167,0,0,0,40,390,-0.00038,-0.00152,0.00040,120,300,270,60
        0.00074, 0.00026, 502, -0.00049, 0.00026, 0.00001, 2, 52, 52, 2, 114, 0.00101, -0.00036, 0.00013, 4, 0, 0, 4, 0, 0.00000, 0.00074, 0.00026, 0, 0, 0, 0, 386, -0.00001, -0.00091, 0.00012, 0, 136, 80, 24, 108, 0.00114, -0.00075, 0.00011, 10, 0, 0, 65, 240, -0.00049, 0.00091, 0.00039, 36, 0, 144, 12, 648, 0.00093, -0.00159, 0.00086, 70, 120, 20, 30, 846, -0.00103, 0.00283, 0.00077, 0, 48, 36, 24, 528, -0.00102, 0.00247, 0.00144, 0, 0, 60, 30, 698, -0.00322, 0.00577, 0.00167, 0, 0, 0, 40, 498, 0.00076, -0.00152, 0.00040, 120, 300, 270, 60
    };
    double result = FlowPredictor_Predict(ModelName, static_cast<int>(test.size()), test.data());

    std::cout << "Predict: " << result;
    std::getchar();
}

int main()
{
    FlowPredictor_SetLogPath(L"%AppData%\\MetaQuotes\\Terminal\\Common\\Files\\Models\\");
    FlowPredictor_SetModelPath(L"%AppData%\\MetaQuotes\\Terminal\\Common\\Files\\Models\\");
    FlowPredictor_LoadModel(ModelName);
    processOneVector();
    //processCsv("c:\\Users\\atwice\\AppData\\Roaming\\MetaQuotes\\Terminal\\Common\\Files\\Data\\vectorBaseCross-test.csv");
}
