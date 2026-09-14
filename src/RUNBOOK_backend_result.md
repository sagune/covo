# 早上照这个走（后端训练结果判读 runbook）

> 一句话：**先看 `MORNING_REPORT.txt` 的 headline，再核对三个闸门，最后按 §3 的档位决定下一步。**
> 全过程 5 分钟；所有命令都是复制粘贴可跑的。

---

## 0. 30 秒确认它跑完了

```bash
cd /root/autodl-tmp
ls .dsh_checks/rerank/TRAIN_PLAN2_DONE .dsh_checks/rerank/SEQUENCER_DONE   # 两个都应存在
tail -3 .dsh_checks/rerank/sequencer.log                                   # 看有没有 REFUSING/TIMED OUT
bash .dsh_checks/preflight.sh | tail -3                                    # 应报 0 problem(s)
```
若 `TRAIN_FAILED` 存在：训练失败，`MORNING_REPORT.txt` 里已有诊断，**不要**在它之上继续跑任何臂。

## 1. headline（唯一要记住的数字）

```bash
head -40 .dsh_checks/rerank/MORNING_REPORT.txt
```
它给出：新 adapter 在 **ST-CMDS held-out** 上的 `restore[deployable]` **CER / 召回 / destroyed**、
与 **4.9356%** 基线的差、以及是否达标（`<= 4.50%`）。
（EARLY 版在迁移评测一出来就有；完整版在对照臂跑完后**覆盖**同一文件。）

## 2. 三个闸门（CER 单独好看不算数）

| 闸门 | 现在（V-C 基线） | 必须 | 跌破就回退 |
|---|---:|---|---|
| **编辑精确率** | **78.5%** | **≥ 70%** | **< 54% ⇒ 必然净亏**（§10.12） |
| 召回 | 95.98% | 不低于 95.93% | 下降即回退 |
| destroyed | 0 | **保持 0** | 非 0 即回退 |
| JSON 解析失败率 | **0.000%** | 保持 0 | 非 0 即格式退化 |

编辑精确率在 `MORNING_REPORT.txt` 里直接给出（`EDIT PRECISION`）。
**为什么它是第一闸门**：ST-CMDS 上后端收益的 **57% 来自编辑**（§10.6），
而同一个模型在编辑率 37% 时会掉到 51.6% 精确率并**净亏**（§10.12）。

## 3. 按 headline 分档

| ST-CMDS CER | 判读 | 下一步 |
|---|---|---|
| **≤ 4.50%** | ✅ 达标 | 写进论文（`PAPER_TABLES_rerank.tex` 表 7/9/10 之后加一行）；核对三个闸门后即可定稿这一节 |
| 4.50–4.75% | 方向对、幅度不够 | 先看报告里的**合并臂**（生成器编辑 + 选择器补放弃行，§10.16，已自动算）；仍不够 → DPO（sequencer 已自动跑过） |
| 4.75–4.94% | 没吃到选择误差 | 看**似然打分臂**：若它也 ≈ 前端，说明"9B 看不见这个选择"→ 如实写负结果 |
| ≥ 4.94% | 没赢现 adapter，或训练把它推成了"乱改" | 看编辑精确率：< 54% 就是过度编辑 → 用 keep-it 加权 DPO 重跑 |

**哪些备用方案当天做得到**：似然打分臂 ~1 h、合并臂 0 GPU、DPO ~3.5 h（都已自动跑过）。
`epoch2` / `selector` 各还要 ~7.5 h 训练，**当天来不及**——不要把它们排进当天计划。

## 4. 若要人工重跑某一条（幂等）

```bash
# 全部产物都存在就跳过；绝不单独启动对照臂（会抢 GPU）
pkill -f run_night_sequencer.sh; sleep 70          # 等陈旧锁释放
cd /root/autodl-tmp && setsid nohup bash .dsh_checks/run_night_sequencer.sh \
  < /dev/null >> .dsh_checks/rerank/sequencer.out 2>&1 & disown
ps -eo args | grep -c '^bash .dsh_checks/run_night_sequencer.sh$'   # 必须是 1
```

## 5. 红线（引用前必须成立）

- 训练集**不含** ST-CMDS / THCHS 任何 uttid：已核验**完全 reference 重叠 0/0、uttid 重叠 0/0、
  共享 8-gram 0**（扫描 581,719 条训练文本，§10.5）。**引用这组数字时不要只引用"我以为查过了"。**
- 出厂 adapter 的下游数字是**接口外**测得的（§10.2），引用时必须标注；
  可以说"修复消掉 60.2% 的下游伤害"，**不能说**"修复不损害下游"。
- 论文表 3 里 ST-CMDS"放宽 + 修复重排"那一格**仍是 `\TBD`**（两条前端口径差约 1 个编辑，不可混填，§0.1）。

## 6. 判读时的三个已知口径陷阱（都踩过一次）

1. **raw vs `restore[deployable]`**：所有进结论的表都在 deployable 上；`edit_precision.py` 默认已改对（§10.14）。
2. **"池外行数"有两个定义**：685（要求 `e_out>0`）vs 711（真·ref∉pool）；论文用 **711**（§10.11）。
3. **"更好候选"有可见/全池两种**：训练集上 61.4%（可见）vs 73.4%（全池）；引用要写明（§10.13）。
