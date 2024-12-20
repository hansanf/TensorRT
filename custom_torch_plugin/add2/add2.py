import os
# # 设置当前工作路径
# os.chdir('custom_torch_plugin/add2')

# # 确认当前工作路径是否更改成功
# print("Current working directory:", os.getcwd())

import time
import numpy as np
import torch
from torch.utils.cpp_extension import load

cuda_module = load(name="add2",
                   sources=["add2.cpp", "add2.cu"],
                   verbose=True)
 
# c = a + b (shape: [n])
n = 1024 * 1024
a = torch.rand(n, device="cuda:0")
b = torch.rand(n, device="cuda:0")
cuda_c = torch.rand(n, device="cuda:0")
 
ntest = 10
 
def show_time(func):
    times = list()
    res = list()
    # GPU warm up
    for _ in range(10):
        func()
    for _ in range(ntest):
        # sync the threads to get accurate cuda running time
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        r = func()
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()
 
        times.append((end_time-start_time)*1e6)
        res.append(r)
    return times, res
 
def run_cuda():
    cuda_module.torch_launch_add2(cuda_c, a, b, n)
    return cuda_c
 
def run_torch():
    # return None to avoid intermediate GPU memory application
    # for accurate time statistics
    a + b
    return None
 
print("Running cuda...")
cuda_time, _ = show_time(run_cuda)
print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))
 
print("Running torch...")
torch_time, _ = show_time(run_torch)
print("Torch time:  {:.3f}us".format(np.mean(torch_time)))






'''
model 输入两个参数的用法：
forward symbolic接口都设置对应的形参
torch.onnx.export 中模型的输入用tuple 将两个输入包含起来
'''

import torch
import torch.nn as nn
from torch.autograd import Function

class MyAdd(Function):
  @staticmethod
  def forward(ctx, input_a, input_b, n):
    # return nn.GELU()(input+ add_num)
    cuda_c = torch.rand(n, device="cuda:0")
    cuda_module.torch_launch_add2(cuda_c, input_a, input_b, n)
    return cuda_c
  
  '''
  RuntimeError: No Op registered for MyGelu with domain_version of 11

  ==> Context: Bad node spec for node. Name: /MyGelu OpType: MyGelu
  op name要加命名空间，如果在 g.op() 里不加前面的命名空间，则算子会被默认成 ONNX 的官方算子。
  '''
  @staticmethod
  def symbolic(g, input_a, input_b, size):
    return g.op("test::MyAdd", input_a, input_b, n_i=size)

class TwoMyAdd(nn.Module):
  def __init__(self):
    super(TwoMyAdd, self).__init__()

  def forward(self, x, y):
    x = MyAdd.apply(x, y, 100)
    return x

if __name__ == "__main__":
  model = TwoMyAdd()
  # (batch_size, channel_size, height, width)
  x = torch.full((1024,), 1.0, device="cuda:0")
  print(f"input : {x}")
  out = model(x, x)
  print(f"model out: {out}")

  input_name_a = "a_input"
  input_name_b = "b_input"
  output_name = "xx_output"
  # 动态shape 输入/输出	
  # dynamic_axes = {input_name: {0: 'batch_size', 2: 'input_height', 3: 'input_width'}}
  # 只需要指定 input 为 dynamic, output是根据input 自动推算的
  # dynamic_axes = {input_name_a: {0: 'batch_size'}}									
  # print(f"dynamic_axes: {dynamic_axes}")
  with torch.no_grad():
    torch.onnx.export(
        model,
        (x, x),
        "custom_add2.onnx",
        opset_version=11,
        input_names=[input_name_a, input_name_b],
        output_names=[output_name],
        # dynamic_axes=dynamic_axes
    )
