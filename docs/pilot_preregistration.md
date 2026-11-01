# FrcSub 100 人 pilot 预注册：GPT-4o mini v1

锁定日期：2026-11-01。机器可读设计为 `configs/pilot_frcsub_gpt4omini_v1.yaml`。本文件在正式学生样本生成前写入；运行后不得根据结果修改模型、提示、样本量、预算或关卡界值。

## 当前状态

设计已锁定，代码仅完成离线测试与 dry run。**尚未执行预检，尚未生成正式学生样本，也未判定任何 pilot 关卡通过。** 执行前需要在当前进程环境中提供 `OPENAI_API_KEY`；密钥不得写入 YAML、运行清单、GitHub 或论文附件。

## 模型与解码

- 模型：`gpt-4o-mini-2024-07-18`，使用固定快照而非浮动别名。
- 接口：OpenAI Chat Completions，`service_tier=default`。
- 学生作答：`temperature=0`、无状态、每题一次独立请求。
- 输出：JSON object，`max_completion_tokens=1024`，请求 `seed=42`；seed 只作复现辅助，不声称跨后端完全确定。
- 模型文档：<https://platform.openai.com/docs/models/gpt-4o-mini>。
- 选择理由：非推理型、支持显式 temperature、具有固定快照，适合把解码设置作为受控因素。
- 范围说明：这是闭源模型方法 pilot，不替代正式全实验中两个不同家族或规模开源模型的要求。

MiniMax-M3 与 MiniMax-M3.1-Flash 不进入该正式 pilot。MiniMax 官方接口文档说明两者固定 `temperature=1`，传入其他值不会生效；因此不能把 temperature 0 的请求记录为实际解码条件。文档：<https://platform.minimax.cn/docs/api-reference/text-chat-openai>。既有 M3 输出仍只作 smoke。

## 四单元与样本

| 单元 | 人设 | 提示策略 | 虚拟学生 | 题目 | 目标请求 | 单元累计上限 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `none_answer_first` | 无人设 | 先答后解释 | 100 | 20 | 2000 | 2200 |
| `none_error_examples` | 无人设 | 错误示例 | 100 | 20 | 2000 | 2200 |
| `attribute_answer_first` | 属性人设 | 先答后解释 | 100 | 20 | 2000 | 2200 |
| `attribute_error_examples` | 属性人设 | 错误示例 | 100 | 20 | 2000 | 2200 |

所有单元使用原题、相同数据切分种子 42、相同 DINA 估计设置（30 组初值、300 次迭代上限）和 200 次真人 split-half 噪声重复。属性人设从真人拟合半的 DINA 掌握模式先验抽样；留出真人数据不用于生成人设或拟合基线。主分析保留全部 20 题，item_008 只允许另行做有依据的剔除敏感性。

目标学生请求共 8000 次。另做 2 次合成算术预检，不计入学生样本。**全局累计硬上限为 8800 次，包含预检、成功请求、解析失败、传输失败和续跑请求。** 每个单元 2200 次上限提供至多约 10% 的失败重试余量，但不能通过续跑重置。所有尝试先写本地 JSONL 账本；配置在执行后改变时拒绝续跑。

## 提示策略

两种策略共享以下要求：以指定学生身份作答，不追求助手式正确率；先给最终答案，再可选给简短演算；只返回 `{"final_answer": "...", "work": "..."}`；计分只使用最终答案。

`answer_first` 不加入额外错误提示。`error_examples` 在相同基础提示中加入以下固定英文文本：

> Use these only as examples of mistakes a student can make, not as a demand to err:
> adding numerators and denominators directly, or failing to regroup when subtracting
> a fraction from a whole number. If such a mistake follows from this student's profile,
> reproduce it; otherwise respond as this student would.

动态人设、题目、历史字段与完整提示模板以 `src/student_fidelity/simulation/simulators.py` 为准。此次 pilot 全为无状态，因此历史字段固定为 `No previous answers.`。题目顺序继续使用循环拉丁方平衡。

## 预检与执行

先运行不访问网络、不读取密钥的 dry run：

```bash
PYTHONPATH=src python scripts/run_locked_pilot.py \
  --config configs/pilot_frcsub_gpt4omini_v1.yaml
```

正式执行必须显式加 `--execute`：

```bash
PYTHONPATH=src python scripts/run_locked_pilot.py \
  --config configs/pilot_frcsub_gpt4omini_v1.yaml \
  --execute
```

执行器先用合成题 `Compute 1/2 + 1/4.` 分别请求 temperature 0 和 1。只有实际返回模型与锁定快照完全一致、`finish_reason=stop` 且答案 JSON 可解析时才继续。失败会占用一次预算并停止；只有显式重新执行才会续跑。预检通过不代表模型输出质量或任何研究关卡通过。

真实数据、预检输出、学生原始输出和运行结果仅保留在本地 `runs/`，不上传 GitHub。

## 生成后的分析顺序

1. 检查四单元请求完整性、失败率、模型快照、finish reason 和 token usage。
2. 对每个单元使用 `scripts/audit_run.py`，显式指定 `--dimension-b-model dina`、30 组初值、300 次迭代上限。
3. 关卡 3 使用 `dina_mean_nonmaster_excess`，全部 20 题未加权均值，界值 0.30；至少一种提示策略达到界值才满足该关卡的通过规则。
4. 长度敏感性只作报告，不参与提示选择；模式层回读继续禁用，属性层只分析四个预定属性。
5. 关卡 1 当前仍未通过，拟合半回读天花板的切分敏感性必须作为局限保留，不能因 LLM 结果而改写历史诊断。
6. 只有完成上述审计后，才讨论是否进入正式全实验；8000 次 pilot 不自动授权后续扩量。
