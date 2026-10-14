# 无padding逐候选条件似然：新锁定协议

用户2026-10-05明确授权“那就无padding呗，继续”。旧batch失败协议和单条padding修补审计完整保留，不修改。新目录candidate_likelihood_single_20261005。

模型固定旧最佳B/text最终2162步adapter，不训练、不加载连接器、不更改已安装库。所有候选统一batch=1、无padding、全1attention_mask、position_ids=0..L-1、use_cache=False，每次只输入一个候选目标；两候选分开前向。没有安装padding补丁，因为此路径不需要padding清零。不能将旧跨路径偏移解释成已消失，只是不再混用路径。

原190个输入/旧预测产生的候选对、180句保持byte-identical；提示、候选顺序hash、SenseVoice封存CTC字段、完整JSON目标、逐token条件logp之和及精确平分选A均不变。主分数不长度归一化、不设置声学阈值、不恢复原文。每组均保留1334句、21102参考字、599热词mentions。

none/true/zero/mismatch四证据条件，加候选顺序反转，共8组，每组190局部决定和1334完整预测，共3040目标单条前向。固定原序为主结果；反序只作稳健性对照，不挑最好顺序或调整权重。参考仅在8组分数封存之后用于评分诊断。

本地/远程6项单元测试。运行前adapter tensor精确一致、4条旧文本raw重放一致；首4输入候选对重复同一路径，反转执行顺序后所有token logp必须精确一致；首条trimmed/full logits差<=0.005nat验证因果位置。不再比较异构batch/padding路径，用户已授权改变该项协议。检查异常保留停止，不自动改范围、模型、提示、分数或恢复。

全量报告CER、热词召回、S/D/I、正确输入误改、54有害/98有效/38中性块选择、配对95%CI和原序/反序敏感性。目标CER不恶化且召回提高，真实证据优于无证据/清零/错配。仍是重复历史AISHELL contextual dev诊断，不是官方14326全dev或新heldout。候选字符等长/token不一定等长、强制二选一、全句语言概率、字典/参考书写疑点与控制分数OOD均注明。
