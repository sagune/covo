# 后端（COVO LoRA）训练计划 — 无人值守夜间运行

**目标**：把 ST-CMDS held-out 的端到端 CER 从 **4.9356%** 推向 **4.5%**，且召回不劣化（当前 95.98%，已等于池 oracle）、保护跨度破坏不增加（当前 0）。

**为什么打后端**：前端已到顶。学习排序器前端 5.2722%，其上界由 24 个束搜索特征锁死在 ~5.19%（同池训练+同池评估 5.1885% ≈ 折外 5.2027%，非过拟合），而池 oracle 是 1.9372%——剩下 **3.25pp 是选择误差，束特征看不见**。选择正是后端的职责，而后端目前只贡献 0.34pp。

---

## 一、数据集选择与理由

| 候选 | 行数 | 需要重新解码 | 协议 | 决策 |
|---|---:|---|---|---|
| **AISHELL train** | **17,301** | **否**——候选池已在盘上（`src/logs/cb_sensevoice_w14_train_evidence_full.jsonl.gz`，100% 带 16 候选、字段齐全） | **干净**：ST-CMDS/THCHS 保持一次性迁移 | ✅ **采用** |
| AISHELL train 子集 | 5,000 | 否 | 干净 | 备用 |
| ST-CMDS train（旧界面现成 SFT） | 138,180 | 否，但提示词是老格式，与推理界面不匹配 | 域内 | 否 |
| ST-CMDS train（新界面） | 95,418 | **是，约 53h 解码** | 域内 | 否 |

**去重后 17,301 行，0 重复**（`id` 在分片内是位置索引，需按 `(split,id)` 去重）。其中：
- reference 在池内 **88.0%**
- **池内存在比 top-1 更好的候选 73.4%（12,706 行）** ← 这就是"后端必须选得更准"的训练信号

---

## 二、训练设计

**起点**：现有 AISHELL adapter `qwen35_9b_aishell_hardneg_dropout_2epoch_20260815/checkpoint-30022`（"在当前模型的基础上"）。

**界面**：与推理臂**逐字一致**——同一个桥接、同一组 flag（`--rank-mode transmitted`）。

> **【2026-09-15 00:45 更新：这一点已从推测升级为核实事实】**
> 读出厂 adapter 的**原始训练文件第一行**（`interface_probe2.py`）后确认，界面差异是**四处同时改变、其中两处把目标反过来了**：
>
> | | 训练时看到的（`train.cb-hardnegative.qwen.jsonl`，138,180 行） | 本会话推理时喂的 |
> |---|---|---|
> | system | `…保守的中文ASR后纠错器…`**`证据不足时保持第一候选`**（100 字符） | `…候选选择器…`**`不要默认保守复制 top-1`**（165 字符） |
> | user | `N-best文本：`/`N-best拼音：` 纯列表（682 字符） | `ASR top-1:`/`Protected hotwords`/`reliability labels`/`Stable spans`/`Hotword evidence`（1260–2651 字符） |
> | 结尾 | `请输出：{"text":"纠错后的完整句子"}` | `请输出 JSON：{"text":"…"}` |
> | max_length | **1024**（p99 提示词 1631 ⇒ 当年 31% 的行被截断） | 2048 |
>
> AISHELL adapter 是同族的 127 字符保守版本，同一个矛盾。
> 这**直接解释了 ST-CMDS 上 93.3% 的 `no_change`**（见 §七 E40）：adapter 忠实地执行了它学到的那条规则。
> 所以 **4.9356 是接口外（out-of-interface）测出来的数字**，衡量的是"一个保守纠错器被要求当选择器用"，
> **不是 COVO 的选择能力上限**；把它训到逐字一致的界面上是最小、最必要的修正实验。

**目标**：`target = normalized reference`，序列化为推理时完全相同的 `{"text":"..."}`。
**同一个目标自动训练两种能力**，按 reference 是否在池内分流：
- reference ∈ pool（88.0%）→ 目标是池内某条 ⇒ **选择**
- reference ∉ pool（12.0%）→ 目标是池里没有的文本 ⇒ **额外修正/编辑**

**绝不限制输出词表**，所以编辑能力不被禁止。唯一会扼杀编辑能力的是"输出候选编号"的选择器方案——**不做主线，仅作对照消融**（可把"选择"与"编辑"的贡献干净拆开）。

**超参**（照抄现有配方，仅把 LR 调低因为这是续训）：
```
LoRA r=16, alpha=32, dropout=0.05, target=q,k,v,o,gate,up,down_proj
max_length=2048（我们的提示词 p99=1631，1024 会截断 31% 的候选列表）
epochs=1, LR=2e-5 (linear), warmup=3%
batch 4 × accum 2, bf16, gradient checkpointing, --disable-thinking
save_steps=500
```

---

## 三、时间估算（有实测锚点）

**锚点**（现有那次训练的真实日志 `train_runtime = 1.985e5 s`）：
```
138,180 行 × 2 epochs = 34,546 步 × 5.75 s/step = 55.1 小时
= 1.392 samples/s（batch 4 × accum 2 = 每步 8 条，序列平均 ~371 token）
```
**我们的提示词平均 1015 token（2.3×）** → 单步约 13 s。
```
1 epoch = 17,301 / 8 = 2,163 步 × 13 s ≈ 7.8 小时
```
**±30% 误差**（"序列变长→单步线性变慢"是外推）。**phase 1 的 20 步试点会把它换成实测值**。

| 阶段 | 内容 | GPU |
|---|---|---|
| 0 | 建 SFT 数据 + 评测提示词（CPU） | 0（已完，3.4 min） |
| 1 | 20 步计时试点 | ~10 min |
| 2 | 全量训练 1 epoch | ~8 h |
| 3 | 按 AISHELL dev 端到端 CER 选检查点（最后 2 个 + final） | ~30 min |
| 4 | ST-CMDS + THCHS-30 迁移评测 | ~2.2 h |

**合计约 11 小时**，夜间完成。

---

## 四、我会监控什么（自主进行，不打扰你）

**阶段标记**：`P2_PHASE0_DONE` → `P2_PHASE1_DONE` → `P2_PHASE2_DONE` → `P2_PHASE3_DONE` → `P2_PHASE4_DONE` → `TRAIN_PLAN2_DONE`；失败写 `TRAIN_FAILED` 并**立即停止**（v1 的教训：不设硬门就会级联空跑）。

**日志**：`/root/autodl-tmp/.dsh_checks/rerank/train_run_v2.log`、`pilot_out.log`、`train_out.log`。

**四个防自欺指标**：
1. `edited` 行数——训练后暴涨说明学到"总是改"
2. JSON 解析失败率——格式退化
3. **`beyond-N-best` 分区的 imp/wor**（现在 **94 / 22**，占后端总增益一半）——这是**编辑能力的仪表盘**，掉到 imp<20 就是"用编辑换了选择"，**立即回退**
4. 始终与现有 adapter 在同一条 message 文件上对照（ST-CMDS 的对照就是已有的 4.9570）

**泄漏红线**：训练数据只含 AISHELL train；已用 uttid 集合核验 `splits/train.jsonl` 与 ST-CMDS held-out test **重叠 0**。

---

## 五、失败预案

| 情况 | 应对 |
|---|---|
| phase 1 试点失败 | 立即停，看 `pilot_out.log`，不烧 8 小时 |
| 训练 OOM | 降 batch 到 2、accum 到 4；或 qlora |
| 训练完但 AISHELL dev 没涨 | 判定"界面校准"无收益 → 转 **DPO**（chosen=reference / rejected=top-1，数据零成本） |
| ST-CMDS 涨但 AISHELL 掉 | 以 AISHELL 为调优集，判定过拟合 → 降 epoch 或加混合 |
| 编辑能力掉 | 回退，改用在 R_out 行上用"池内最优候选"当硬负例的 DPO 变体 |

---

## 六、诚实预期

**不承诺 4.5。** 后端现在只贡献 0.34pp，池 oracle 说还有 3.0pp 的选择空间，但训练能吃下多少未知。这一版的主要价值是三点：(1) 修掉"界面错配"这个已知缺陷；(2) 第一次用**新界面的监督**教选择；(3) 无论涨跌，都能给出"后端能不能吃掉选择误差"的**确定性答案**。

按用户要求：**我先做「全量训练」这一版**（1 epoch、全 17,301 行），不预先做子集筛选。

---

## 七、夜间新增（2026-09-15 00:30–00:50）：解剖结论 + 早上照此判读

### 7.1 错误解剖（E40）：ST-CMDS 的病是**欠编辑**，不是改错

`covo_error_anatomy.py`（`--policy deployable` 能逐位复现 4.9356/3.1674/2.5874，所以两个视角不会各自漂移）：

| | ST-CMDS V-C | THCHS-30 | AISHELL dev |
|---|---:|---:|---:|
| `no_change` 行占比 | **93.3%** | 63.2% | 72.0% |
| 净编辑（字符） | −336 | −15 | −231 |
| `miss_visible`（更好候选就在提示词里却被跳过） | **1322 行 / 2.7046pp** | 644 / 1.1671pp | 178 / 1.0662pp |
| 其中落在 `no_change` 里的 | **1198 行 / 1341 字符** | — | — |
| 净损失（`sel_loss`+`edit_loss`） | 74 字符 | 532 字符 | 97 字符 |

⇒ ST-CMDS 上可回收价值 (2.39pp) 是当前净损失 (0.13pp) 的 **18 倍**；同一族 adapter 在 THCHS-30 改 36.8% 的行、
在 ST-CMDS 只改 6.7%——**5 倍编辑倾向差 = 校准差异，不是能力差异**（原因见 §二 的更新）。

### 7.2 4.5 的账（用 `gain_split.py` 精确化）

出厂 COVO 在 ST-CMDS 上的 +189 字符收益里，**43% 来自选择（+82）、57% 来自编辑（+107）**，
而选择只实现了选择空间的 **6.6%**（82 / 1251）。THCHS-30 上选择这一侧甚至是**负的（−130）**，
整体收益全靠编辑（+145）兜底——这是"它根本没在做选择"的直接证据。

- 目标还需 **+245 字符**。若编辑保持 +107 不变，**选择必须做到 +327**。
- **分母约定（统一）**：选择可回收空间 = 输入误差 − 可见候选最优误差 = **1169 字符**；已实现 **+82 = 7.0%**；
  目标要求 **327/1169 = 28.0%** ⇒ 选择的实现率要从 **7.0% 提到 28%（4 倍）**。
  （等价写法 82/(82+1169)=6.6% 与 327/1496=21.9%，本计划统一用前者。）
- 全池上界 **1.9390%**（比当前输出低 2.9966pp = 1683 字符）⇒ 目标只需要全池可用空间的 **26%**。
- 训练集"池内有更好候选"占 **73.4%**，ST-CMDS 评测只占 **25.8%** ⇒ 训练**必然**把模型推向"更敢改"，
  正好对着 ST-CMDS 缺的方向。**风险不对称，且偏向目标方向。**（这是 §9.10 "过度编辑"风险的镜像。）
- **编辑是当前收益的一半以上，所以"编辑不被牺牲"是硬约束**：`beyond-N-best` 的 imp/wor 必须守住（现 94–98 / 22），
  否则即使 CER 好看也是在拆收益来源。反过来，THCHS-30 的选择侧是负的，
  说明"选择变好"与"编辑不变"在同一目标下**不冲突**。

### 7.3 早上照此判读（判定表）

看 `MORNING_REPORT.txt`（EARLY 版约 09:20 就有，含 headline 与闸门指标）、
`compare_old_vs_new.txt` 与 `TRAIN_SUMMARY.txt`，按 ST-CMDS `restore[deployable]` 分档：

| ST-CMDS 结果 | 判读 | 下一步 |
|---|---|---|
| **≤ 4.50%** | 达标。再确认召回 ≥ 95.93%、`destroyed` = 0、编辑精确率 ≥ 70% | 写进论文（§4.2 的表），跑 `run_vd_arm.sh` 的 VD 提示词臂做接口敏感性 |
| 4.50–4.75% | 方向对、幅度不够 | 看 §10.16 的**合并臂**（生成器编辑 + 选择器补放弃行，报告里已自动算）；再不行转 DPO |
| 4.75–4.94% | 训练没吃到选择误差 | 看似然打分臂：若它也 ≈ 前端，说明"9B 看不见这个选择"，DPO 也难救 → 如实报负结果 |
| ≥ 4.94%（没赢现 adapter） | 界面校准假设被推翻 **或**训练把模型推成了"乱改" | 看 `edited` 是否暴涨、编辑精确率是否 < 54%；按 §五 回退 |
| AISHELL dev 涨、ST-CMDS 不涨 | 只学到 AISHELL 特性，未泛化 | 以 AISHELL 为唯一调优集的纪律要求如实记录，**不改测试集** |

**关于备用方案的现实成本（09-15 更新）**：`run_train_next.sh` 的三个模式里，
`epoch2` 与 `selector` 各自还要 **~7.5 h** 训练，**当天来不及**；真正可行的是
**似然打分臂（~1 h，已接进 sequencer）**、**合并臂（0 GPU，报告里自动算）**、
以及 **DPO（~3.5 h，sequencer 会按"编辑精确率 < 60% 就用 keep-it 加权数据"自动选数据）**。
次序上 sequencer 已经实现为：未达标 → 似然打分 → DPO → 对照臂。

**必须同时看的三个数（任一异常即回退，不看 CER 单项）**：
`edited` 行数（现 334/5130）、`beyond-N-best` 的 imp/wor（现 **94–98 / 22**，这是编辑能力仪表盘）、
`destroyed`（现 0）与召回（现 95.98%）。

**"编辑能力不能被牺牲"现在是一个统计判据，不是一个点阈值**：`compare_arms.py` 对
**beyond-N-best 分区**（只有生成能帮上忙的 711 行）做配对 bootstrap，直接输出
`d(编辑增益)` 的 95% 区间与 `P(B 的编辑 >= A 的)`。判读规则：

| 输出 | 含义 |
|---|---|
| `P(B's editing >= A's) ≥ 0.9` | 编辑能力守住了 ✅ |
| 区间下界 < −0.5pp | **回退条件**（脚本会显式打印 WARNING） |
| `imp` 掉到 < 20 | 按 §9.5 的原始阈值回退 |

参考值（现 adapter 内部对照 V-A vs V-C）：`d(edit gain) +0.0856 pp [−0.0489, +0.2385]`，
`P = 0.912` —— 即连两个只差提示词字段的臂之间，这个量本身就是有噪声的，
所以**必须看区间而不是看单点**。

### 7.4 训练损失轨迹（判读结果时的第二个维度）

| step | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 |
|---|---|---|---|---|---|---|---|---|---|
| loss | 1.865 | 1.322 | 0.672 | 0.321 | 0.261 | 0.245 | 0.239 | 0.232 | **0.223** |
| grad_norm | 0.93 | 0.57 | 0.32 | — | — | — | — | — | 0.114 |

**怎么读这个数**：目标 token 数约 16（`{"text":"..."}` + 句子），所以 0.223 nats/token 对应
序列级完全正确的概率大约 `e^(-0.223×16) ≈ 3%`——**这不是"已经背下来了"，而是"学到一个还不错的策略"**。
所以：
- loss 在 step ~120 之后基本走平，说明**格式与复制行为**很快就学会了；
  剩下 2000 步（LR 线性降到 0）是在细调**该复制哪一条**。这与"选择"是难点、格式不是难点一致。
- **若最终 loss 仍在 0.15–0.25**：模型没有完全拟合训练目标（训练集上"池内有更好候选"占 73.4%），
  说明它学到的是温和的偏离策略。这种情况下 ST-CMDS 涨幅预计在 **0.1–0.3pp** 量级，
  大概率**到不了 4.5**，应直接转 DPO。
- **若最终 loss 掉到 <0.08**：它高度拟合了训练目标，ST-CMDS 上有希望拿到更大的选择实现率，
  但同时要警惕 `edited` 暴涨（学到了"总是在候选里挑一个"）。
- 无论哪种，**判据仍是 ST-CMDS 的 `restore[deployable]` + 三个守门指标**，损失只是解释用的第二维度。

### 7.5 排队中的臂 —— 已改为**单一 GPU 所有权**（`run_night_sequencer.sh`）

**为什么改**：原来四个臂各自"等上一个的标记"，这在计划固定时没问题，但**结果不好时顺序是错的**——
如果训练没打到 4.5，目标明确要求的 DPO 备用方案会被排在对照（2.5h）+ VD 臂（1.7h）+ 接口臂（0.9h）
之后，白等约 5 小时 GPU，而这三个臂没有一个能改善 CER。所以改成**一个进程按测量结果决定顺序**，
它是唯一启动 GPU 工作的进程：

```
1 报告   训练一结束立刻（纯 CPU）出 TRAIN_SUMMARY.txt
2 补齐   用 get_cer.py 确认头部数字存在；缺了就重跑对应迁移臂
3 判定   读新 adapter 的 ST-CMDS restore[deployable] CER
4 执行   达标 (<=4.50) -> 只跑三个对照臂（原计划）
         未达标        -> 先跑 DPO，再跑三个对照臂
5 选点   若 AISHELL dev 说 final 不是最好，用胜出的检查点重跑 ST-CMDS 迁移
6 早上   写 MORNING_REPORT.txt（一屏给出结论 + 闸门指标 + 所有臂）
```

| 臂 | 脚本 | 作用 | 触发 |
|---|---|---|---|
| 阶段 3/4 | `run_train2.sh` | 选点 + ST-CMDS/THCHS 迁移 | 训练结束 |
| 头报 | `consolidate_train.py` | AISHELL-dev 检查点排名 | 训练结束（CPU） |
| **DPO（仅未达标）** | `run_train_next.sh dpo` | 偏好对教"选对候选而不是复制 top-1" | CER > 4.50 |
| 对照 | `run_controls.sh` | 现 adapter 在**同三份提示词**上跑一遍 | 之后 |
| VD 臂 | `run_vd_arm.sh` | 新 adapter 跑带 `rerank=` 的提示词 | 之后 |
| 接口臂 | `run_oldiface_arm.sh` | 现 adapter 跑**它自己的**训练界面（单变量对照 E41） | 之后 |
| 消歧 | `post_controls_extra.sh` | `compare_adapter_identity.txt`（CPU） | CONTROLS_DONE |

**已把原来四个独立等待进程（run_controls / run_vd_arm / run_oldiface_arm / post_train_watch）
全部停掉**，避免和 sequencer 抢 GPU；它们的逻辑已由 sequencer 按顺序调用，标记链不变。

判读用的两个新工具：`get_cer.py`（从日志里取一条臂的 CER/召回/破坏/imp/wor，取不到就返回 rc=1）、
`compare_arms.py`（配对 bootstrap + 行为迁移 + THE BET/REGRESSION）。

**早上要跑的两条对照命令**（都需要 predictions 文件已存在，CPU）：

```bash
S=datasets/stcmds/cb_sensevoice_heldout/hotword/test; R=.dsh_checks/rerank
PY=/root/autodl-tmp/great/bin/python
# 1) 新 adapter vs 现 adapter：配对 bootstrap + 行为迁移矩阵 + "THE BET / REGRESSION" 两行
$PY .dsh_checks/compare_arms.py --a $R/e2eSTCMDS_VA.predictions.jsonl \
   --b $R/train_aishell_v1/stcmds_final.predictions.jsonl \
   --aligned $S/aligned.txt --uttid $S/uttid \
   --label-a "existing adapter" --label-b "trained adapter"
# 2) 收益拆成选择/编辑两侧，看 4.5 还差多少
$PY .dsh_checks/gain_split.py --records $R/train_aishell_v1/stcmds_final.predictions.jsonl \
   --label "ST-CMDS trained adapter"
```

`compare_arms.py` 的 **THE BET / REGRESSION 两行是这套方案的成败判据**：
"THE BET" = 现 adapter 放着不动、新 adapter 改成对的那些行（欠编辑被修好）；
"REGRESSION" = 现 adapter 本来改对了、新 adapter 又退回不动的那些行。
若 THE BET ≫ REGRESSION → 假设成立；若两者相当 → 训练只是把编辑和选择互相抵消。

### 7.6 数据/产物勘误（避免早上误读）

- `e2eSTCMDSFIX` = **放宽准入 + FIXED 重排**，实测 **5.6443% / 93.60 / 0**。
  （我曾在一时命令里把它标成"relax+手工修复 5.1279"——**错**。5.1279 = 放宽 + **出厂**重排；5.5214 = **原**准入 + 修复重排。）
- `run_controls.sh` 用 **AISHELL** adapter 跑 ST-CMDS 提示词，却标注成 "ST-CMDS, existing adapter"；
  而同报告正文把 4.9570（**ST-CMDS** adapter）称作 "the control above"——**两个不同 adapter**。
  已用 `compare_adapter_identity.txt` 消歧。
