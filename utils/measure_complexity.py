import torch
import time
import copy  # <--- 1. 记得导入 copy
from thop import profile

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DANCER, UNet, ACDAE, DACNN


def measure_model(model, input_shape=(1, 2, 256), device="cuda", runs=1000):
    model.to(device)
    model.eval()

    # 创建假数据
    dummy_input = torch.randn(input_shape).to(device)

    # ===================================================
    # 修正点：使用深拷贝的模型来计算 FLOPs
    # 这样 thop 的钩子只会挂在 model_copy 上，不会污染原 model
    # ===================================================
    print(f"Calculating FLOPs for {model.__class__.__name__}...")
    try:
        model_copy = copy.deepcopy(model)
        macs, params = profile(model_copy, inputs=(dummy_input,), verbose=False)
        flops = macs * 2
        # 用完销毁，防止内存占用
        del model_copy
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        params = 0
        flops = 0

    # ===================================================
    # 3. 计算推理时间 (使用原始的、干净的 model)
    # ===================================================
    print(f"Warm-up {model.__class__.__name__}...")

    # 这里的 model 是干净的，没有被 thop 污染
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / runs * 1000

    print(f"Model: {model.__class__.__name__}")
    print(f"  - Input Shape: {input_shape}")
    print(f"  - Params:      {params / 1e6:.4f} M")
    print(f"  - FLOPs:       {flops / 1e6:.4f} M")
    print(f"  - Latency:     {avg_time_ms:.4f} ms")
    print("-" * 40)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}\n")

    models_to_test = [UNet(), ACDAE(), DACNN(), DANCER()]

    test_shape = (1, 2, 256)

    for m in models_to_test:
        measure_model(m, input_shape=test_shape, device=device)
