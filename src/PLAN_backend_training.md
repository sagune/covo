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

**界面**：与推理臂**逐字一致**——同一个桥接、同一组 flag（`--rank-mode transmitted`）。这一点是关键：现 adapter 是用**老提示词格式**训的（平均 371 token），而我们推理用的是 selector + 热词证据 + 保护 + 名次自洽（平均 1015 token）。**它的 4.9356 是"没按当前界面训过"的 adapter 跑出来的**，校准界面本身可能就有白捡的收益。

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
