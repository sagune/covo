# 当前服务器源码快照

当前保留的最终方法是[五数据集基础 9B 纠错系列](results/mainline-multidataset.md)。本快照中的神经连接器、局部匹配和后续声学复核源码只作为历史尝试留痕，不再作为默认模型入口。为保持源码哈希和实验依赖，整理结果时没有移动或删除快照内的文件。

GitHub 原仓库以早期 CoVo 为主；服务器包含整个 CBSense 工程、嵌套 CoVo 包和多轮锁定实验。本次不覆盖旧脚本，在 `server_snapshot` 保存相同相对目录结构，保留实验引用关系。

原始根目录 `/root/autodl-tmp`，原 Git HEAD `402b364`（完整 SHA 见 `SOURCE_MANIFEST.json`）。导出的是当前文件系统，不仅是历史 commit；清单记录每个原文件的路径、大小和 SHA256。

源码包 SHA256：`9fc75d555ea107b4ed35b7bf6114dad130c2655234c96c47691a2a569820f9a1`。

## 后端版本

- `covo/` 为服务器主工程后端。
- `cbwhisper_covo_migration_20260609_tar_extracted/covo/` 为迁移实验引用的运行时代码。只收集源码，不包括原约 62 GB 的模型/数据。两版本的 `infer_lora_text.py` 哈希不同，不能悄悄替换。
- 仓库根目录旧 CoVo 包仍保留。安装同名包须隔离环境，或明确 `PYTHONPATH`，避免误导入旧版。

## 实验与重现

`src/experiments/` 收集实验目录顶层源码、说明、锁定协议与部分小型摘要/审计。未导出嵌套运行目录、依赖 overlay、输入样本、逐条预测、模型缓存或 checkpoint。因此可用于阅读、归档和准备重现，不代表历史脚本能在新机器直接执行。

原协议绝对路径、数据哈希、提示词、检查点记录保持原样。新运行应另建目录/协议并准备自己的资源，不应修改封存协议冒充历史实验。报告中的本地输出路径是原留痕位置；未发布预测与数据需另按授权取得，不能因此声称已独立复算。

## 轻量检查

```bash
# 根目录历史后端
PYTHONPATH=src python -m unittest discover -s tests
# 从 server_snapshot 执行主工程后端检查
PYTHONPATH=covo/src python -m unittest discover -s covo/tests
```

单元测试和语法检查不验证 GPU 数值一致性、重载精确性或完整 CER。正式训练/推理仍需要锁定实验的 smoke 和完整验证门槛。

本次发布不修改远端 Git 工作区，不启动训练/推理，不提交连接信息或认证材料。热图为机制示意，真实例句来源单独记录。
