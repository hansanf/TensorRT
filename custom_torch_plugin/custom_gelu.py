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
	x = torch.full((3,4), 1.0)
	print(f"input : {x}")
	out = model(x)
	print(f"model out: {out}")
	with torch.no_grad():
		torch.onnx.export(
				model,
				x,
				"custom_gelu.onnx",
				opset_version=11,
				input_names=['xx_input'],
				output_names=['xx_output'])
