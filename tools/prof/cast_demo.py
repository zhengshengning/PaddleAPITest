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

# paddle.cast(Tensor([2, 1, 32768, 32768],"float16"), dtype=Dtype(float16), )

# paddle.Tensor.cast(Tensor([128256, 4096],"float16"), Dtype(float16), )
# paddle.Tensor.cast(Tensor([3584, 152064],"float16"), Dtype(float16), )
# paddle.Tensor.cast(Tensor([152064, 3584],"float16"), Dtype(float16), )

# m = 2
# n = 1
# k = 32768
# l = 32768
# test_loop = 5170

m = 152064
n = 3584
test_loop = 6064
# numpy_tensor = (numpy.random.random([m, n, k, l]) - 0.5).astype("float16")
numpy_tensor = (numpy.random.random([m, n]) - 0.5).astype("float16")
paddle_x, torch_x = init_input(numpy_tensor)
numel = (numpy_tensor.size)
# test_loop = 2147483647 * 20 // numel
print("numel=", numel , "test_loop=", test_loop)


# paddle_out = paddle.cast(paddle_x, 'float16')
paddle_out = paddle.Tensor.cast(paddle_x, 'float16')

with paddle.no_grad():
    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
    start = time.time()
    for i in range(test_loop):
        paddle.Tensor.cast(paddle_x, 'float16')
    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
    end = time.time()
    timeused = end - start
    print("paddle forward", timeused)

numpy_tensor = (numpy.random.random(paddle_out.shape) - 0.5).astype("float16")
paddle_grad, torch_grad = init_input(numpy_tensor)

try:
    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
    start = time.time()
    for i in range(test_loop):
        paddle.grad([paddle_out], [paddle_x], grad_outputs=paddle_grad, allow_unused=True)
    paddle.base.core._cuda_synchronize(paddle.CUDAPlace(0))
    end = time.time()
    timeused = end - start
    print("paddle backward", timeused)
except Exception as e:
    print(f"paddle 反向失败")



torch_out = torch_x.to(torch.float16)
print(torch_out.shape)
with torch.no_grad():
    torch.cuda.synchronize()
    start = time.time()
    for i in range(test_loop):
        torch_x.to(torch.float16)
    torch.cuda.synchronize()
    end = time.time()
    timeused = end - start
    print("torch forward", timeused)
try:
    torch.cuda.synchronize()
    start = time.time()
    for i in range(test_loop):
        torch.autograd.grad([torch_out], [torch_x], grad_outputs=torch_grad, retain_graph=True)
    torch.cuda.synchronize()
    end = time.time()
    timeused = end - start
    print("torch backward", timeused)
except Exception as e:
    print(f"torch 反向失败")
