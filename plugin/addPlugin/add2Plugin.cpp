/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
// #if CUDA_VERSION >= 10010

#include <cstring>
#include <vector>

#include "NvInfer.h"
#include "common/bertCommon.h"
#include "common/serialize.hpp"
#include "add2Plugin.h"

using namespace nvinfer1;
using namespace nvinfer1::plugin;
using namespace nvinfer1::plugin::bert;

namespace
{
char const* const kADD2_PLUGIN_VERSION{"1"};
char const* const kADD2_PLUGIN_NAME{"CustomAddPluginDynamic"};
} // namespace

// Static class fields initialization
PluginFieldCollection Add2PluginDynamicCreator::mFC{};
std::vector<PluginField> Add2PluginDynamicCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(Add2PluginDynamicCreator);

Add2PluginDynamic::Add2PluginDynamic(const std::string name, const DataType type)
    : mLayerName(name)
    , mType(type)
{
    // mHasBias = (bias.values != nullptr);
    // if (mHasBias)
    // {
    //     void* cudaMem{nullptr};
    //     PLUGIN_CUASSERT(cudaMalloc(&cudaMem, getWeightsSize(bias, mType)));
    //     PLUGIN_CUASSERT(cudaMemcpy(cudaMem, bias.values, getWeightsSize(bias, mType), cudaMemcpyHostToDevice));
    //     make_cuda_shared(mBiasDev, cudaMem);
    // }
}

Add2PluginDynamic::Add2PluginDynamic(const std::string name, void const* data, size_t length)
    : mLayerName(name)
{
    gLogVerbose << "Add2PluginDynamic deserialize\n";
    deserialize_value(&data, &length, &mType);
    // deserialize_value(&data, &length, &mLd);
    // deserialize_value(&data, &length, &mHasBias);

    // if (mHasBias)
    // {
    //     PLUGIN_VALIDATE(mLd > 0);
    //     char const* d = static_cast<char const*>(data);
    //     make_cuda_shared(mBiasDev, deserToDev<char>(d, mLd * getElementSize(mType)));
    // }
}
// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* Add2PluginDynamic::clone() const noexcept
{
    try
    {
        gLogVerbose << "Add2PluginDynamic clone\n";
        auto* plugin = new Add2PluginDynamic(*this);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

nvinfer1::DimsExprs Add2PluginDynamic::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
    int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(outputIndex == 0);
        return inputs[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool Add2PluginDynamic::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 1);
        // 2 inputs, 1 outputs, so 3 input/output in total
        PLUGIN_VALIDATE(0 <= pos && pos < 3);
        PLUGIN_VALIDATE(inOut != nullptr);
        auto const* in = inOut;
        auto const* out = inOut + nbInputs;
        bool const consistentFloatPrecision = (in[0].type == in[pos].type);
        switch (pos)
        {
        case 0:
            return in[0].type == DataType::kFLOAT && in[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
        case 1:
            return in[1].type == DataType::kFLOAT && in[1].format == PluginFormat::kLINEAR && consistentFloatPrecision;
        case 2:
            return out[0].type == DataType::kFLOAT && out[0].format == PluginFormat::kLINEAR
                && consistentFloatPrecision;
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return false;
}

void Add2PluginDynamic::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    gLogVerbose << "Add2PluginDynamic configurePlugin\n";

    try
    {
        PLUGIN_VALIDATE(in != nullptr);
        PLUGIN_VALIDATE(nbInputs == 2);
        PLUGIN_VALIDATE(nbOutputs == 1);
        PLUGIN_VALIDATE(mType == in[0].desc.type);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

size_t Add2PluginDynamic::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

template <typename TDataType>
int32_t Add2PluginDynamic::enqueueTyped(
    void const* input_a_, void const* input_b_, void* output_, int32_t const inputVolume, cudaStream_t stream) noexcept
{
    TDataType const* input_a = static_cast<TDataType const*>(input_a_);
    TDataType const* input_b = static_cast<TDataType const*>(input_b_);
    TDataType* output = static_cast<TDataType*>(output_);
    int32_t const cols = inputVolume / mLd;
    int32_t const rows = mLd;

    return launch_add2_kernel(output, input_a, input_b, 100); //TODO
}

int32_t Add2PluginDynamic::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    try
    {
        PLUGIN_VALIDATE(inputDesc != nullptr);
        PLUGIN_VALIDATE(inputs != nullptr);
        PLUGIN_VALIDATE(outputs != nullptr);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return STATUS_FAILURE;
    }

    int32_t const inputVolume = volume(inputDesc[0].dims);

    // Our plugin outputs only one tensor.
    // Launch CUDA kernel wrapper and save its return value.
    switch (mType)
    {
    case DataType::kFLOAT: return enqueueTyped<float>(inputs[0], inputs[1], outputs[0], inputVolume, stream);
    // case DataType::kHALF: return enqueueTyped<half>(inputs[0], outputs[0], inputVolume, stream);
    default: return STATUS_FAILURE;
    }
}

// IPluginV2Ext Methods
nvinfer1::DataType Add2PluginDynamic::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        PLUGIN_VALIDATE(index == 0);
        PLUGIN_VALIDATE(inputTypes != nullptr);
        PLUGIN_VALIDATE(inputTypes[0] == DataType::kFLOAT);
        return inputTypes[0];
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DataType{};
}

// IPluginV2 Methods

char const* Add2PluginDynamic::getPluginType() const noexcept
{
    return kADD2_PLUGIN_NAME;
}

char const* Add2PluginDynamic::getPluginVersion() const noexcept
{
    return kADD2_PLUGIN_VERSION;
}

int32_t Add2PluginDynamic::getNbOutputs() const noexcept
{
    return 1;
}

int32_t Add2PluginDynamic::initialize() noexcept
{
    gLogVerbose << "Add2PluginDynamic initalize\n";
    return 0;
}

void Add2PluginDynamic::terminate() noexcept
{
    gLogVerbose << "Add2PluginDynamic terminate\n";
}

size_t Add2PluginDynamic::getSerializationSize() const noexcept
{
    // const size_t wordSize = getElementSize(mType);
    // const size_t biasSize = mHasBias ? mLd * wordSize : 0;
    // return sizeof(mType) + sizeof(mHasBias) + sizeof(mLd) + biasSize;
    return sizeof(mType);
}

void Add2PluginDynamic::serialize(void* buffer) const noexcept
{
    serialize_value(&buffer, mType);
    // serialize_value(&buffer, mLd);
    // serialize_value(&buffer, mHasBias);
    // if (mHasBias)
    // {
    //     PLUGIN_ASSERT(mLd > 0);
    //     char* d = static_cast<char*>(buffer);
    //     serFromDev(d, static_cast<char*>(mBiasDev.get()), mLd * getElementSize(mType));
    // }
}

void Add2PluginDynamic::destroy() noexcept
{
    gLogVerbose << "Add2PluginDynamic destroy\n";
    // This gets called when the network containing plugin is destroyed
    // mBiasDev.reset();
    delete this;
}

void Add2PluginDynamic::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* Add2PluginDynamic::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

///////////////

Add2PluginDynamicCreator::Add2PluginDynamicCreator()
{
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    // mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 1));

    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* Add2PluginDynamicCreator::getPluginName() const noexcept
{
    return kADD2_PLUGIN_NAME;
}

char const* Add2PluginDynamicCreator::getPluginVersion() const noexcept
{
    return kADD2_PLUGIN_VERSION;
}

PluginFieldCollection const* Add2PluginDynamicCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* Add2PluginDynamicCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        gLogVerbose << "Add2PluginDynamicCreator createPlugin\n";
        PLUGIN_VALIDATE(fc != nullptr);

        // Weights bias{DataType::kFLOAT, nullptr, 0};
        int32_t typeId = -1;
        plugin::validateRequiredAttributesExist({"type_id"}, fc);

        for (int32_t i = 0; i < fc->nbFields; i++)
        {
            PLUGIN_VALIDATE(fc->fields[i].name != nullptr);
            std::string fieldName(fc->fields[i].name);
            if (fieldName.compare("type_id") == 0)
            {
                typeId = *static_cast<int32_t const*>(fc->fields[i].data);
            }
            // // onnx 模型中的 attributes, filedName在 builtin_op_importers.cpp中进行定义
            // if (fieldName.compare("bias") == 0)
            // {
            //     bias.values = fc->fields[i].data;
            //     bias.count = fc->fields[i].length;
            //     bias.type = fieldTypeToDataType(fc->fields[i].type);
            // }
        }

        if (typeId < 0 || typeId > 3)
        {
            gLogError << "Add2PluginDynamicCreator: invalid typeId " << typeId << std::endl;
            return nullptr;
        }

        return new Add2PluginDynamic(name, static_cast<DataType>(typeId));
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* Add2PluginDynamicCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call Add2PluginDynamic::destroy()
    try
    {
        return new Add2PluginDynamic(name, serialData, serialLength);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void Add2PluginDynamicCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    try
    {
        PLUGIN_VALIDATE(libNamespace != nullptr);
        mNamespace = libNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* Add2PluginDynamicCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// #endif // CUDA_VERSION >= 10010
