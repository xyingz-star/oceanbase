# q_kmeanspp

CUDA k-means++（fp32 与 int8 量化距离两种），**numpy 入参 / 出参**，调用方式接近「装好后 `import` 即用」：

```python
from q_kmeanspp import kmeans_pp_l2, kmeans_pp_l2_int8
import numpy as np

x = np.random.randn(10000, 768).astype(np.float32)
c = kmeans_pp_l2(x, 256, cosine=False)  # 默认 int8 量化距离
c_fp32 = kmeans_pp_l2(x, 256, cosine=False, int8=False)  # 全 fp32 距离
# 与默认等价：kmeans_pp_l2_int8(x, 256)
```

## 安装

需要本机有 `nvcc`（CUDA toolkit），运行期需要 `numpy`。

```bash
cd test/q_kmeanspp
pip install --user -e .   # 无权限写系统目录时加 --user，或用 venv
```

`setup.py` / `make` 会在 `q_kmeanspp/libq_kmeanspp_cuda.so` 生成动态库。

仅编译 `.so`（不 pip，适合无 numpy / 只测 CUDA）：

```bash
make -C test/q_kmeanspp
```

开发时也可不安装，把 `test` 的父目录加入 `PYTHONPATH`，并先 `make` 生成 `libq_kmeanspp_cuda.so`。

## 指定 `.so` 路径

```bash
export Q_KMEANSPP_CUDA_SO=/path/to/libq_kmeanspp_cuda.so
# 兼容旧名：
export OB_EXTERNAL_KMEANS_PP_CUDA_SO=/path/to/libq_kmeanspp_cuda.so
```

## 源码

`csrc/kmeans_pp_l2.cu` 为唯一实现源。`oceanbase/tools/cuda_kmeans_pp/Makefile` 会通过 `wildcard` 查找该文件（若 `oceanbase` 是符号链接且不在同一棵目录树，会回退到 `$(HOME)/test/q_kmeanspp/...`，也可手动 `make CU=/绝对路径/.../kmeans_pp_l2.cu`）。
