import torch
import paddle
import numpy
import time
import os

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
# os.environ["FLAGS_use_system_allocator"] = "0"
# os.environ["FLAGS_share_tensor_for_grad_tensor_holder"] = "1"
paddle.framework.set_flags({"FLAGS_use_system_allocator": False})
paddle.framework.set_flags({"FLAGS_share_tensor_for_grad_tensor_holder": True})

device = torch.device("cuda:0")
torch.set_default_device(device)
paddle.device.set_device('gpu:0')


def init_input(numpy_tensor):
    paddle_x = paddle.to_tensor(numpy_tensor)
    torch_x = torch.tensor(numpy_tensor, requires_grad=True)
    paddle_x.stop_gradient = False

    numpy.testing.assert_allclose(
        paddle_x.numpy(),
        torch_x.cpu().detach().numpy(),
        1e-10,
        1e-10,
        err_msg='intput diff'
    )
    return paddle_x, torch_x

# paddle.mode(Tensor([2, 10, 10],"float64"), -1, )

m = 2
n = 10
k = 10
dim = 1
test_loop = 1278
numpy_tensor = (numpy.random.random([m, n, k]) - 0.5).astype("float64")
paddle_x, torch_x = init_input(numpy_tensor)
numel = (numpy_tensor.size)
# test_loop = 2147483647 * 20 // numel
print("numel=", numel , "test_loop=", test_loop)


paddle_out = paddle.mode(paddle_x, dim, keepdim=True)

with paddle.no_grad():
    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
    start = time.time()
    for i in range(test_loop):
        paddle.mode(paddle_x, dim, keepdim=True)
    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
    end = time.time()
    timeused = end - start
    print("paddle forward", timeused)

numpy_tensor = (numpy.random.random(paddle_out[0].shape) - 0.5).astype("float64")
paddle_grad, torch_grad = init_input(numpy_tensor)

try:
    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
    start = time.time()
    for i in range(test_loop):
        paddle.grad([paddle_out[0]], [paddle_x], grad_outputs=paddle_grad, allow_unused=True)
    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
    end = time.time()
    timeused = end - start
    print("paddle backward", timeused)
except Exception as e:
    print(f"paddle 反向失败")



torch_out = torch.mode(torch_x, dim, keepdim=True)
print(torch_out[0].shape)
with torch.no_grad():
    torch.cuda.synchronize()
    start = time.time()
    for i in range(test_loop):
        torch.mode(torch_x, dim, keepdim=True)
    torch.cuda.synchronize()
    end = time.time()
    timeused = end - start
    print("torch forward", timeused)
try:
    torch.cuda.synchronize()
    start = time.time()
    for i in range(test_loop):
        torch.autograd.grad([torch_out[0]], [torch_x], grad_outputs=torch_grad, retain_graph=True)
    torch.cuda.synchronize()
    end = time.time()
    timeused = end - start
    print("torch backward", timeused)
except Exception as e:
    print(f"torch 反向失败")
