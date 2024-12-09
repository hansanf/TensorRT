import torch
import torch.nn as nn
from torch.autograd import Function

class MyGelu(Function):
	@staticmethod
	def forward(ctx, input, add_num):
		return nn.GELU()(input+ add_num)
	
	'''
	RuntimeError: No Op registered for MyGelu with domain_version of 11

	==> Context: Bad node spec for node. Name: /MyGelu OpType: MyGelu
	op name要加命名空间，如果在 g.op() 里不加前面的命名空间，则算子会被默认成 ONNX 的官方算子。
	'''
	@staticmethod
	def symbolic(g, input, add):
		return g.op("test::MyGelu", input, add_num_f=add)

class TwoMyGelu(nn.Module):
	def __init__(self):
		super(TwoMyGelu, self).__init__()

	def forward(self, x):
		x = MyGelu.apply(x, 2.5)
		return x

if __name__ == "__main__":
	model = TwoMyGelu()
	# (batch_size, channel_size, height, width)
	x = torch.full((1,3,4,5), 1.0).reshape(-1, 3,4,5)
	print(f"input : {x}")
	out = model(x)
	print(f"model out: {out}")

	input_name = "xx_input"
	output_name = "xx_output"
	# 动态shape 输入/输出	
	# dynamic_axes = {input_name: {0: 'batch_size', 2: 'input_height', 3: 'input_width'}}
  # 只需要指定 input 为 dynamic, output是根据input 自动推算的
	dynamic_axes = {input_name: {0: 'batch_size'}}									
	print(f"dynamic_axes: {dynamic_axes}")
	with torch.no_grad():
		torch.onnx.export(
				model,
				x,
				"custom_gelu.onnx",
				opset_version=11,
				input_names=[input_name],
				output_names=[output_name],
				dynamic_axes=dynamic_axes
		)
