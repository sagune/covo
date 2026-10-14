# 同一真实样本的来源

来源：outputs/joint-lora-20260923/final_audited/D/dev/text.predictions.jsonl，id / audio_id：BAC009S0724W0461，condition：text。

本轮只读取留存结果做图，不进行指标重算或新的模型推理。D 是原始第一轮 9B 检查点的文本对照，不是声学复核模型。

- 参考：京华时报记者王海欣摄
- CBSense Top-1：金华时报记者王海欣摄
- N-best 第 2 条：京华时报记者王海欣摄
- N-best 第 3 条：金华时报记者王海欣说
- 其余候选：金华日时报记者王海欣摄；津华时报记者王海欣摄；金华时报记者王海欣社；今华时报记者王海欣摄；精华时报记者王海欣摄。
- 原始 9B prediction：京华时报记者王海欣摄
- 原始 model_output：{"text":"京华时报记者王海欣摄"}
- 本例检出热词：王海欣，score=0.9992989301681519，图中显示 0.9993。
- 留存候选 1/2 发音字符串：jin hua shi bao ji zhe wang hai xin she / jing hua shi bao ji zhe wang hai xin she。图中仅节选前四个音节。

CBSense 图保留其真实 Top-1 的“金华”错误，不把最终正确结果归功于前端。CEGER 图使用最终“京华”结果。候选排序不按参考重排，未编造正确候选。

CEGER 图是基础训练机制图，使用本条验证样本作说明，不表示该样本参与训练，也不表示初始第一轮训练输入带 CBSense 热词。图内发音与片段摘要属于展示用整理，并非新增模型预测；前三候选片段共识不冒充全 8 条的真实提示共识。候选增强与证据丢弃图标仍是方法示意。

大模型模块名称从 CoVo 改为 CEGER；CBSense 名称保持不变。本轮按用户指定只更新最新 CBSense 总图与第一轮 LoRA 图，不改写旧历史版本。
