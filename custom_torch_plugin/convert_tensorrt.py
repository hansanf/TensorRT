import ctypes
import argparse
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

fp32_tensors = ["/ReduceMax"]
out_tensors = ["/ReduceMax"]
fp32_layer_types = [trt.LayerType.SHUFFLE]

def register_plugins(plugin_library_path):
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    ctypes.CDLL(plugin_library_path)

# 初始化 TensorRT 日志记录器

def print_layer_info(engine):
    """Prints information about the layers in the engine."""
    print("Layer information:")
    for i in range(engine.num_layers):
        layer = engine.get_layer(i)
        print(f"Layer {i}: {layer.name}, {layer.type}, {layer.precision}, {layer.get_output_type(0)}")
        for j in range(layer.num_inputs):
            tensor = layer.get_input(j)
            if tensor is not None:
                print(f"  Input {j}: {tensor.name}, shape={tensor.shape}, dtype={tensor.dtype}")
        for j in range(layer.num_outputs):
            tensor = layer.get_output(j)
            if tensor is not None:
                print(f"  Output {j}: {tensor.name}, shape={tensor.shape}, dtype={tensor.dtype}")    

def convert_trt(args):
    # 创建 TensorRT 构建器
    builder = trt.Builder(TRT_LOGGER)

    # 创建网络定义
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # 创建 ONNX 解析器
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 加载 ONNX 模型
    success = parser.parse_from_file(args.onnx_path)
    if not success:
        print(f"load onnx:{args.onnx_path} failed")
        exit(1)
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        out_flag = False
        set_fp32 = False
        if args.fp16:
            for fp32_tensor in fp32_tensors:
                if fp32_tensor == layer.name:
                    set_fp32 = True
                    print(f"{layer.name} to fp32")
                    layer.precision = trt.float32  # 设置为 FP32
        for out_tensor in out_tensors:
            if out_tensor in layer.name:
                # and "MatMul" not in layer.name and "Concat_66" not in layer.name:
                out_flag = True

        for j in range(layer.num_outputs):
            output_tensor = layer.get_output(j)
            if set_fp32:
                layer.set_output_type(j, trt.DataType.FLOAT)
            if output_tensor is not None and out_flag:
                print(f"  Output {i}: {output_tensor.name}, shape={output_tensor.shape}, dtype={output_tensor.dtype}")
                # output_tensor.name = f'layer_{i}_output_{j}'
                if "Constant" not in output_tensor.name:
                    network.mark_output(output_tensor)

    config = builder.create_builder_config()
    config.max_workspace_size = 2 << 30  # 1 GB
    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    # 构建 TensorRT 引擎
    config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
    # config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    config.set_flag(trt.BuilderFlag.REFIT)
    engine = builder.build_engine(network, config)
    # print_layer_info
    # 序列化引擎
    with open(args.trt_path, 'wb') as f:
        f.write(engine.serialize())

def args_parser_proc():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trt_path",
        default="./model.engine", help="dir for trt engine path"
    )
    parser.add_argument(
        "--plugin_path",
        default="libtrtplugins.so", help="plugin path"
    )
    parser.add_argument(
        "--onnx_path",
        default="./model.onnx", help="dir for onnx path"
    )
    # action="store_true"：表示当命令行中传递了 --fp16 参数时，其值会被设置为 True；如果没有传递该参数，则保持默认值 False。
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="convert fp16"
    )
    return parser
if __name__ == "__main__":
    args_parser = args_parser_proc()
    args = args_parser.parse_args()
    # 注册自定义插件库
    register_plugins(args.plugin_path)

    convert_trt(args)
