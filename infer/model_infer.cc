#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id

using namespace nvinfer1;

// stuff we know about the network and the input/output blobs
const char* INPUT_BLOB_NAME = "xx_input";
const char* OUTPUT_BLOB_NAME = "xx_output";
static Logger gLogger;


void doInference(IExecutionContext& context, float* input, float* output, const int output_size, const int in_size) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);

    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    int mBatchSize = engine.getMaxBatchSize();

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], in_size * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size*sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, in_size * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(1, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

bool isDynamic(Dims const& m) {
  bool b = m.nbDims <= 0;
  for (int i = 0; !b && (i < m.nbDims); ++i) b = m.d[i] < 0;
  return b;
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    const std::string engine_file_path {argv[1]}; // custom_gelu.trt
    std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    auto in_dims = engine->getBindingDimensions(inputIndex);
    printf("input is dynamic shape: %d\n", isDynamic(in_dims));
    if (isDynamic(in_dims)) {
      using Opt = nvinfer1::OptProfileSelector;
      int nb_binding = engine->getNbBindings();
      int nb_profile = engine->getNbOptimizationProfiles();
      int bPerProfile = nb_binding / nb_profile;
      int ip = inputIndex / bPerProfile;
      std::cout << "nb binding: " << nb_binding << " nb profile: " << nb_profile << " ip: " << ip << std::endl;
      in_dims = engine->getProfileDimensions(inputIndex, ip, Opt::kMAX);
      context->setBindingDimensions(inputIndex, in_dims);
      assert(context->allInputShapesSpecified() == true);
      assert(context->allInputDimensionsSpecified() == true);
    }
    auto in_size = 1;
    for (int j=0;j<in_dims.nbDims;j++) {
        std::cout << "in dim " << j << ": " << in_dims.d[j] << std::endl;
        in_size *= in_dims.d[j];
    }
    std::cout << "in_size: " << in_size << std::endl; 

    // dynamic input 的 shape 被set 后，output shape 也会自动被计算
    // 注意，这里只能用 context->getBindingDimensions, 当用 engine->getBindingDimensions 时返回的还是未被计算的 dynamic output shape
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    auto out_dims = context->getBindingDimensions(outputIndex);
    printf("output is dynamic shape: %d\n", isDynamic(out_dims));
    // output 不能是 dynamic shape
    assert(isDynamic(out_dims) == false);
    auto output_size = 1;
    for(int j=0;j<out_dims.nbDims;j++) {
        std::cout << "out dim " << j << ": " << out_dims.d[j] << std::endl;
        output_size *= out_dims.d[j];
    }
    std::cout << "output_size: " << output_size << std::endl; 
    static float* prob = new float[output_size];

    float* blob = new float[in_size];
    for(int j=0;j<in_size;j++) {
      blob[j] = 1.0;
    }

    auto x = engine->getNbBindings();
    std::cout << "nb bindings: " << x << std::endl;

    // run inference
    auto start = std::chrono::system_clock::now();
    doInference(*context, blob, prob, output_size, in_size);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // print output
    // for(int j=0;j<output_size;j++) {
    //   std::cout << prob[j] << std::endl;
    // }

    // delete the pointer to the float
    delete[] blob;
    delete[] prob;
    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}