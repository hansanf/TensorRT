#include <torch/extension.h>
#include "add2.h"
 
void torch_launch_add2(torch::Tensor &c,
                       const torch::Tensor &a,
                       const torch::Tensor &b,
                       int n) {
    launch_add2_kernel((float *)c.data_ptr(),
                       (const float *)a.data_ptr(),
                       (const float *)b.data_ptr(),
                       n);
}
 
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_add2",
          &torch_launch_add2,
          "add2 kernel warpper");
}