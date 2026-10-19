# π0.5 官方 LIBERO 权重评测与 FluxVLA FT 对照

结果汇总日期：2026-10-16；当前三组评测均于 2026-10-15 完成。

**当前官方 checkpoint 的评测结果采用以下两组：FT ZMQ 随机协议 + 预测/执行 10/5，成功率 92.80%；同协议 + 10/10，成功率 95.80%。** 两组并列保留，引用时必须注明执行步长。

这里的“官方”指使用 Physical Intelligence 发布的 `pi05_libero` 权重，不表示采用了 OpenPI 官方评测的全部默认配置。

## 1 当前结果及 FT 对照

每组使用 8 张 A800 80GB，每卡 3 个模型副本和 3 个仿真 worker。Spatial、Object、Goal、Long 各 10 个任务，每任务使用固定初态 0–49；每套 500 个 episode，总计 2,000 个。动作步数上限依次为 220、280、300、520，成功即结束，达到上限仍未成功计失败。平均分为四套成功率的算术平均，不采用 Pass@10，不包括 LIBERO-90。

| Suite | 当前官方 10/5 | 当前官方 10/10 | 本次 FT ZMQ 10/10 | 官方 10/10 − FT（百分点） |
| --- | ---: | ---: | ---: | ---: |
| Spatial | 97.8%（489/500） | 99.2%（496/500） | 99.0%（495/500） | +0.2 |
| Object | 95.8%（479/500） | 98.4%（492/500） | 99.6%（498/500） | −1.2 |
| Goal | 95.0%（475/500） | 96.4%（482/500） | 98.0%（490/500） | −1.6 |
| Long | 82.6%（413/500） | 89.2%（446/500） | 94.2%（471/500） | −5.0 |
| **平均成功率** | **92.80%** | **95.80%** | **97.70%** | **−1.90** |
| **成功次数** | **1,856/2,000** | **1,916/2,000** | **1,954/2,000** | **少 38 次** |

官方 10/10 比 10/5 多成功 60 次，其中 Long 多 33 次。官方 10/10 比 FT 少成功 38 次，其中 Long 少 25 次。每个配置只有一次完整运行，不据此宣称统计显著性。

FT 对照是本次重新运行四套 `full_finetune_bs64` 权重的 **97.70%**，不是 [FluxVLA 公开主表](https://github.com/FluxVLA/FluxVLA#performance) 的 97.95%。公开主表参考仍为 98.6 / 99.6 / 98.0 / 95.6±1.0；主表的历史权重选择、重复次数及 `±` 定义未完整确认。

## 2 模型和当前协议

官方模型来自 `gs://openpi-assets/checkpoints/pi05_libero`，已经由官方针对 LIBERO 微调；本轮没有额外训练。发布说明见 [OpenPI Fine-Tuned Models](https://github.com/Physical-Intelligence/openpi#fine-tuned-models)。同一套官方权重用于四个 suite；FT 使用四套各自微调的权重，来源与固定版本见 [FT 模型准备](libero.md#pi05-ft-model-preparation)。

| 设置 | 当前官方 10/5 | 当前官方 10/10 | 本次 FT ZMQ |
| --- | --- | --- | --- |
| 预测 / 执行动作数 | 10 / 5 | 10 / 10 | 10 / 10 |
| 环境 seed / 稳定化步数 | 0 / 10 | 0 / 10 | 0 / 10 |
| 模型 seed 编号 | `7 + suite 内全局 episode 索引` | 同左 | 同左 |
| 模型 RNG | 每次请求重置，`sequence_noise=False` | 同左 | 同左 |
| 参数存储精度 | FP32，保留官方精度适配 | 同左 | BF16 |
| 模型与仿真调度 | fixed | fixed | dynamic |
| 预处理 / 归一化 | 官方图像、prompt、状态分位数与动作变换 | 同左 | FT 配置对应的图像和 mean/std 变换 |
| 双视角视频 | 4 fps | 2 fps | 2 fps |

完整 suite 的 seed 为 `7 + task_index × 50 + episode_index`，每个 suite 从头编号。当前协议每次请求都重置到该 episode 的 seed，再采样第一份初始噪声；在现有确定性预处理下，同一 episode 各次重规划使用相同初始噪声张量，但 observation 不同，因此 action 仍会变化。它不同于 FT 原生 `scripts/eval.py` 的连续 RNG，也不同于 OpenPI JAX 的默认 RNG。

请求中的 `policy_seed`、`environment_seed` 和 `environment_case_index` 是不同身份字段。
LIBERO 按 case index 选择每任务的固定初态，实际 `simulator_seed=0`；不能把
policy seed 或记录中的 environment seed 当作传给 LIBERO 的模拟器种子。

相对同版本的官方配置，10/5 只覆盖 `themis.server.runner.sequence_noise=False`。10/10 额外将 `eval.eval_chunk_size` 和 `themis.evaluation.execute_horizon` 改为 10，视频帧率按重规划频率改为 2。两组都保留 `keep_params_fp32=True`、官方预处理和 fixed 调度。`evaluation_protocol.json` 与 `official_reference_configs/` 是当时外部任务工具生成的归档，未随本仓库分发；不是当前入口自动生成的文件。

三组虽然对齐了模型随机数协议，FT 与官方仍有权重、预处理、参数精度和调度差异；分差不能直接解释成额外微调的净收益。相同的 episode 身份和 fixed 映射也不保证 GPU 位级确定性。

## 3 耗时和 GPU 统计

| 耗时与吞吐 | 当前官方 10/5 | 当前官方 10/10 | 本次 FT ZMQ |
| --- | ---: | ---: | ---: |
| 评测入口总耗时 | 41 分 01 秒 | 29 分 19 秒 | 40 分 54 秒 |
| 加载与准备累计（约） | 8 分 51 秒 | 8 分 58 秒 | 11 分 53 秒 |
| Rollout 累计（约） | 31 分 43 秒 | 19 分 57 秒 | 28 分 26 秒 |
| Rollout 吞吐（episode/小时） | 3,784 | 6,013 | 4,219 |

上述计时来自评测入口日志，不包含平台排队、拉取镜像及入口前的依赖/哈希检查。Rollout 是各 suite 墙钟时间之和，不累加并行 worker 耗时。阶段时长根据日志秒级时间及四舍五入后的吞吐反推，标为近似值。

官方 10/10 相比 10/5，入口总耗时减少约 28.5%，rollout 吞吐提升约 59%。两组官方单卡峰值均约 59,427 MiB；FT 约 25,287 MiB。GPU 数据为整卡采样，部分 `nvidia-smi` 请求超时，三组汇总均有采样缺口，不能作为无缺失的性能剖析。

## 4 结果目录、版本与验证范围

三组实际源码均为 Themis `ee4257c3c8b2a17926d37683981f6e47be626453`、VLA `f77e1ce1ee5c388ccf5cd906b1573c89f97b437b`。启动日志记录 robosuite 1.5.1、MuJoCo 3.2.6、LIBERO 0.1.0；推理 Torch 2.8.0+cu128，仿真 Torch 2.8.0+cpu。源码 SHA 对应当时运行版本，不代表后续工作区修改已通过评测。

两组官方任务启动时对同一组预期 SHA256 完成校验，并写出 `checkpoint_hashes.json`：

| 文件 | SHA256 |
| --- | --- |
| `model.safetensors` | `598451c937e8839b5a15fa54c35a233dc1cefd633c560d25d9fcbe7b6cf96a1a` |
| `dataset_statistics.json` | `72ff25e64ccdddea45dc816489682c776e5b4da22b7c75cdc6dd9ff3df72cb8c` |
| `tokenizer/tokenizer.model` | `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6` |

三组云端状态均为 `Succeeded`；完整控制台日志包含四套 `run_end status=finished episodes=500/500`。本次汇总的各套成功数由套件结束时的累计成功率唯一还原，计数相减后得到各套结果。**尚未重新下载和逐条审计这三组的 episode 文件、任务覆盖及视频**。

本报告集中保存评测成绩、模型来源、协议和验证范围。详细统计附件已移到仓库外
归档；原始日志、逐 episode 记录和视频继续由外部评测产物保存。

## 5 复跑参数与配置来源

本次官方模型评测使用上面所列 FluxVLA commit 的
`configs/pi05/libero_official/{SUITE}.py`；需先在配置副本中准备 checkpoint、
statistics 和 tokenizer 的绝对路径。这些配置和参数精度适配属于对应 VLA 版本。

对四个 suite 分别执行。以准备好的 `libero_spatial.py` 为例，先在模型环境中启动
10/5 配置的模型服务：

```bash
bash scripts/distributed_model.sh --fluxvla-root /path/to/FluxVLA \
  --config /prepared/libero_spatial.py \
  --ckpt-path /models/pi05_libero/model.safetensors \
  --gpus 8 --workers-per-gpu 3 --scheduling fixed \
  --cfg-options themis.server.runner.sequence_noise=False \
    eval.eval_chunk_size=5 eval.observation_video.fps=4
```

再在仿真环境中连接该服务：

```bash
bash scripts/run_evaluation.sh --config /prepared/libero_spatial.py \
  --server-endpoint tcp://MODEL_HOST:15555 \
  --gpus 8 --workers-per-gpu 3 --scheduling fixed --full \
  --cfg-options themis.evaluation.execute_horizon=5
```

其余 suite 使用各自的准备配置，模型与仿真独立启动。10/10 将模型的执行步长和
仿真的 `execute_horizon` 都改为 10，模型视频 fps 改为 2；预测长度保持 10。
FT 模型路径与四套配置准备见 [FT 模型准备](libero.md#pi05-ft-model-preparation)。
通用启动参数见 [ZMQ 评测启动指南](../guides/evaluation.md)。
这些命令对应本报告的评测协议；清理后的启动器尚未重新执行云端全量评测，
不能以这里记录的分数代替新版本验证。
