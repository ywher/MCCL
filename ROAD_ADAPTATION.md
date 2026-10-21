# MCCL Semi+Source：五域受控适配

入口为 `mccl_source.py`。原始 `MCCL.py`、DeepLabV3+、VOC/Cityscapes配置和原始实验清单均保留。新入口以独立的 `road/` 包提供 DINOv3-B + ReIN + HRDA，不在运行时导入 UniMatchV2 或 SSDA 仓库。

## 与现有 UniMatchV2+Source 对齐的设置

| Transfer | 源域 | 目标类别 | 最低预算 | 标注图像 |
|---|---|---:|---|---:|
| GTA→Cityscapes | GTA | 19 | 1/64 | 46 |
| SYNTHIA→Cityscapes | SYNTHIA | 16 | 1/64 | 46 |
| Cityscapes→ACDC | Cityscapes | 19 | 1/64 | 25 |
| Cityscapes→MUSES | Cityscapes | 19 | 1/64 | 23 |
| Cityscapes→Mapillary | Cityscapes | 19 | 1/128 | 140 |

- 单GPU，目标有标注、无标注、源域GT分别batch=2，40,000次optimizer更新，seed=0。
- 冻结通用DINOv3-B主干，训练ReIN和HRDA解码器；只加载通用VFM，不加载UDA或TC-ADA权重。
- AdamW、lr=1e-4、decoder lr multiplier=10、warm-up=1500、poly power=1、crop=1024。目标预缩放1024×2048；GTA源域1440×2560，其余源域1024×2048。
- Random和R-DKC逐字节复用现有UniMatchV2清单，公平对照默认`SPLIT_SOURCE=random`。同一GPU批内roll得到另一图像作为CutMix donor，与现有UniMatchV2受控适配一致。
- 16类目标使用`gtFine/train_16cls`及`val_16cls`标签，SYNTHIA使用`GT/LABELS_trainid_16cls`；其他使用19类trainIds。运行时检查标签范围，防止把19类标签直接用于16类模型。
- 每10k步及最后一步验证，采用`eval/model_best_ious.csv`；保留独立的最终值、验证曲线和best iteration。

## MCCL损失与HRDA特征接口

保留两路强增强图像一致性、弱图特征dropout一致性、FKA像素对齐与原型紧致性、SAI自适应噪声及掩码扰动。教师为同一个分割模型的stop-gradient弱预测；未引入额外EMA教师。

令`S=(L_target + λ_source L_source)/(1+λ_source)`，默认`λ_source=1`。总损失为：

```text
0.5 * (S + 0.25*L_strong1 + 0.25*L_strong2 + 0.5*L_weak_fp)
+ 0.1*mean(L_point_alignment1, L_point_alignment2)
+ 0.01*L_prototype
+ w(t)*(mean(L_noise1,L_noise2) + mean(L_drop1,L_drop2))
```

`w(t)`采用官方sigmoid ramp-up平方，最终权重0.01；按40k迭代线性定位进度。源域仅贡献独立GT监督，不进入目标伪标签、原型库或目标CutMix。省略两个source参数即可运行Semi。

HRDA的高分辨率分支为随机局部crop，跨视图直接逐点比较会错位。因此FKA/SAI和弱特征dropout使用**覆盖整张裁剪图的低分辨率context分支、分类器前256维特征**，SAI复用其分类器；监督和强图像一致性仍使用两尺度HRDA融合logits。这是明确的HRDA适配选择，不等同于原版ResNet/DeepLabV3+逐数值复现。原型库按16/19类动态分配，弱/强特征按类别键对应，标签和ignore mask只用nearest缩放。

官方代码检查与修正：

- `prote.py`固定21类、256通道且构造时直接`.cuda()`；新原型库使用注册buffer，按类别数分配并随checkpoint保存。
- 官方Cityscapes配置缺少训练脚本实际读取的`feat_nearest`；新配置明确写0.01，与官方Pascal配置相同。
- 官方计算了逐点cosine alignment，但总损失中该项被注释。新入口按其配置`feat_ali_weight=0.1`启用，支持设为0进行消融。
- 官方按筛选后的类别列表位置配对，可能把不同类别对齐，且`need_update`跨类复用可能跳过后续类更新；新实现逐类配对、独立更新。
- 官方SAI同时开启noise/drop时会再次变换已变换的similarity；新实现从原始similarity分别产生两种扰动。ignore像素不参与这些一致性项；空有效区域返回有梯度的0。
- 官方未标注数据仍打开GT；新loader仅加载图像和几何padding，不读取未标注GT。

这些选择应在后续对照实验中表述为“在共同DINOv3-ReIN-HRDA骨架上的MCCL+Source适配”，不能写为官方ResNet结果的复现。

## 准备与运行

在仓库根目录使用现有`reinpy10`环境，或按`requirements-road.txt`准备PyTorch环境：

```bash
python scripts/prepare_road_data.py \
  --ssda-root /path/to/SSDA --unimatch-root /path/to/UniMatch-V2
python scripts/validate_road_data.py

# 检查命令；CUDA_DEVICES必须显式指定。
TRANSFER=gta2cityscapes CUDA_DEVICES=0 DRY_RUN=1 \
  bash scripts/train_semi_source_hrda.sh

# 单项正式训练，移除DRY_RUN。
TRANSFER=gta2cityscapes CUDA_DEVICES=0 \
  bash scripts/train_semi_source_hrda.sh
```

更换`TRANSFER`即可运行表中五项；需要单独考察共同选样设置时可显式设`SPLIT_SOURCE=rdkc`；主公平对照不使用R-DKC。准备脚本仅创建本仓库数据/权重链接并复制split，不更改源数据、现有UniMatchV2或SSDA结果。清单与哈希在`splits/`对应路径及`splits/road_preparation_manifest.json`。

结果目录：`exp/transfer/<transfer>/mccl_source/dinov3_base_rein_hrda/<selection>/<ratio>/`。检查点包含adapter/decoder、optimizer、迭代数、best iteration、原型库及split哈希；通用冻结VFM需保留。`RESUME=1`恢复同一配置和数据清单，拒绝静默覆盖已有训练。恢复会重建随机数据流，不保证与未中断训练逐位一致。

独立评估使用`python evaluate_road.py --checkpoint /path/to/model_best.pth --output /path/to/ious.csv`。移动主机后可指定`--data-root`和`--pretrained-path`；`--val-id-path`可替换验证清单。

## 验证

```bash
python -m unittest discover -s tests -v
python scripts/smoke_road.py --device cuda:0 \
  --result validation/hrda_19class_smoke.json
python scripts/smoke_road.py --device cuda:0 \
  --config configs/road_hrda_source/synthia2cityscapes.yaml \
  --result validation/hrda_16class_smoke.json
```

smoke只执行一次synthetic optimizer step，随后验证推理及模型/原型保存恢复；为让随机输入覆盖所有损失，**仅smoke临时将两个置信度阈值设为0**，正式配置仍为0.95。它不产生可填入benchmark的精度。完整验证结果与本地真实图像入口检查记录保存在`validation/`，未启动40k正式训练。

已完成的本地核验（2026-10-21）：

| 核验 | 结果 |
|---|---|
| CPU功能测试 | 6项通过；含Semi/Semi+Source监督权重数值检查 |
| 五域Random/R-DKC数据 | 10组全部文件存在，训练/验证图像无交集，清单与参考字节一致 |
| 19类真实HRDA synthetic step | 通过，1024×1024，batch=2，显存峰值8.079 GiB |
| 16类真实HRDA synthetic step | 通过，1024×1024，batch=2，显存峰值7.769 GiB |
| 真实GTA/Cityscapes数据入口 | 1步训练、1张验证、checkpoint加载均通过 |
| 独立评估 | 导出CSV与短测训练时验证完全一致 |
| 启动/语法检查 | 五项dry-run、Python编译及Shell语法通过 |

以上为适配有效性检查；合成短测的显存记录来自本机RTX 3090，不能代替正式训练性能或训练时长。原始验证记录见`validation/summary.json`。

## 代码来源

- MCCL官方clone基线：`2ca68f98ac64243ccc16e9397aa7aed2cf62be1f`。
- 共同HRDA实现及初始五域配置参考本地UniMatchV2：`198a5aacc0e5af45a04960d389edfaabe72568d4`。
- `road/road_vfm/`从已适配的共同实现复制，保留原版权与许可注释；`road/hrda_base.py`仅调整导入位置和类名。`adaptation_provenance.json`记录复制文件哈希。

## 2026-10-21 正式对照排期

主对照使用Random selection seed=0的固定清单，与UniMatchV2逐字节相同；training seed=0。五项分别为46/46/25/23/140张，均40k。`scripts/run_minimum_random_queue.sh`负责单GPU接续；显式前序标记完成且GPU连续60秒无compute进程、显存低于500MiB后启动，任何训练失败会停止本队列。队列不自动删除实验结果。

本轮分配（2026-10-21更新）：EDEN/G3优先运行C2Map；C2Mu已于2026-10-21在新HYY2/G0（RTX 4090）从头启动，数据集与代码已部署。Eden上约900步的C2Mu已按用户要求中断并删除目录与日志，没有checkpoint或最终结果。HYY4/G0在精确benchmark标签及B型VFM权重完成SHA-256核验后运行G2C；MAKKA/G0运行C2A、MAKKA/G1运行S2C，两项各自单卡并行；原HYY/G2及本地G0的MCCL等待队列已撤销，原SSDA E7/E8队列保持。详细实时状态与ETA维护在SSDA的`paper/experiments/tmm_p0_reuse/current_training_status.md`与`MCCL_training_plan.md`，以实查状态为准。

运维约定：后续大文件优先通过用户的OSS传输；传输完成后统一校验一次，日常进度查询不重复计算SHA-256，出现异常时再核验。
