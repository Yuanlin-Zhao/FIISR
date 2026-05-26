# FIISR

代码进行图像超分辨率训练与测试。当前推荐把数据集统一放在项目根目录的 `dataset/` 下，代码目录为 `FIISR/`。

## 数据集下载

请从百度网盘下载以下三个数据集，下载后解压到 `dataset/` 目录中：

| 数据集 | 文件 | 百度网盘链接 | 提取码 |
| --- | --- | --- | --- |
| LLVIPSR | `LLVIPSR.zip` | https://pan.baidu.com/s/11kn7SsRCPWK2CMK8frADoA | `qfki` |
| M3FD | `M3FD.zip` | https://pan.baidu.com/s/1Vz9EpTbGt6MSuQKYSYnTMg | `mru2` |
| IR700 | `IR700.zip` | https://pan.baidu.com/s/1i1kaxKrxAg-FzXKlyaxm7g | `gyna` |

推荐目录结构如下：

```SR
SR/
├── dataset/
│   ├── LLVIPSR/
│   │   ├── train/
│   │   │   ├── GT/
│   │   │   ├── LR/
│   │   └── val/
│   │       ├── GT/
│   │       ├── LR/
│   ├── M3FD/
│   └── IR700/
└── 
```

如果训练或测试其他数据集，需要在对应的 `.yml` 配置文件中修改 `dataroot_gt`、`dataroot_lq`路径。


## 训练

训练配置文件位于：

```
/options/super_resolution/train/train_FIISR_X4.yml。值得注意的是本文的退化是在训练过程中退化的，因此GT和LR是一个路径。我们上传的LR则是为了让其他的网络进行训练。
```




训练输出会保存到：

```SR
experiments/train_FIISR_X4/
```

包括模型权重、训练状态和日志。

## 测试

测试配置文件位于：

```SR
options/super_resolution/test/test_FIISR_S_X4.yml
```

测试前请先修改配置文件中的权重路径：

```yaml
path:
  pretrain_network_g: experiments/train_FIISR_X4/models/net_g_50000.pth
```

测试结果会保存到：

```SR
results/test_FIISR_S_X4/
```

## 切换数据集

以 M3FD 为例，把训练配置中的路径改为：

```yaml
datasets:
  train:
    dataroot_gt: dataset\M3FD\train\GT
    dataroot_lq: dataset\M3FD\train\GT
  val:
    dataroot_gt: dataset\M3FD\val\GT
    dataroot_lq: dataset\M3FD\val\GT
```

IR700 同理，将路径中的 `M3FD` 替换为 `IR700` 即可。

