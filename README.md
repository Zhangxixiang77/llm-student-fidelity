# LLM Student Fidelity

用认知诊断模型检验 **LLM 模拟学生是否“答对了题，却不是同一个学生”**。本仓库把保真度与可控性分开评估：题目通过率吻合只是起点，题目间依赖结构、属性掌握结构和隐含 slip/guess 才是核心证据。

## 当前状态

这是一个可运行的 FrcSub pilot 方法骨架，不是正式实验结果。首版包含：

- 统一的学生 × 题目二元作答与 Q 矩阵接口，严格按学生切分拟合半/留出半；
- 独立 Bernoulli、高斯 copula、2PL IRT、DINA、logit-link G-DINA 五类生成基线；
- 题目通过率相关/RMSE、总分 Wasserstein 距离、四分相关矩阵 RMSE、完整作答模式 TV 距离；
- 真人 split-half 噪声分布与 95% 界值、生成条件 bootstrap CI 函数；
- 属性/能力/无人设，逐题无状态/有状态作答，题目位置循环拉丁方平衡；
- 属性与模式回读、DINA／G-DINA 可达天花板、合格属性限定分析、隐含 slip/guess 与能力泄漏率；
- 留出题目对相关／log-OR 检验、Holm 校正、稀疏单元格诊断和固定 Q 矩阵敏感性；
- DINA 可复现多组随机初值、按训练边际对数似然选优及逐初值收敛审计；
- DINA／G-DINA 参数恢复、SRMSR 与题目对假阳性率模拟，以及全量 DINA 已发表参数对照；
- 重复学生切分诊断，以及有文献依据、但不改变主分析题目集的 item_008 剔除敏感性；
- 在 100 人 pilot 前锁定的 G-DINA 历史配置，以及保留原界值的 DINA 修订版 `configs/pilot_gates.frcsub.dina.v2.yaml`；
- 双人设 × 双提示策略的四单元正式 pilot 预注册、固定提供方模型与解码条件、8800 次全局调用硬上限；
- 离线 DINA mock 和可选 OpenAI-compatible 接口；原始输出逐条保存为 JSONL；
- 分数/混合数/有限小数等价评分、固定温度模型适配和显式断点恢复；
- 一个完全离线、使用合成数据的端到端 demo。

**默认不访问网络、不调用付费模型、不附带 FrcSub 数据。** `demo` 的结果只验证软件流程，不能作为 LLM 证据。

## 快速开始

需要 Python 3.11+。

```bash
python -m pip install -e '.[dev]'
pytest
student-fidelity demo --output-dir runs/demo --students 40 --noise-repeats 10
```

也可以不安装命令行入口：

```bash
PYTHONPATH=src python -m student_fidelity.cli demo --output-dir runs/demo
```

主要产物：

```text
runs/demo/
├── summary.json             # 设计、模型元数据、噪声界值和全部结果
├── metrics.csv              # 各条件主要数值指标
├── raw_simulation.jsonl      # 最终答案、原始输出、finish reason 与 token usage
├── run_manifest.json         # 防止跨配置续跑的数据/设计/模型配置指纹
├── assigned_alpha.csv        # 指定的属性掌握模式
├── split_half_noise.csv
└── generated/               # 模拟器及各基线的二元作答矩阵
```

## 使用真实数据运行一个 pilot 单元

1. 按 [`docs/data.md`](docs/data.md) 准备 `responses.csv`、`q_matrix.csv`，真实模型实验还需要 `items.csv`。
2. 复制并修改 `configs/pilot.example.yaml`。默认仍是离线 mock。
3. 运行：

```bash
student-fidelity pilot --config configs/pilot.example.yaml
```

如改用 `openai_compatible`，在环境变量中设置密钥，不要写入 YAML 或提交 `.env`：

```bash
export OPENAI_API_KEY='...'
student-fidelity pilot --config configs/pilot.real-model.yaml
```

MiniMax 的小样本配置见 `configs/pilot.minimax.yaml`。该配置只用于连通性和流程 smoke；正式 M3 pilot 使用下方独立锁定配置。密钥从环境变量 `MINIMAX_API_KEY` 读取，不发送 `temperature`，并把实际固定值 1.0 记录在元数据中。若请求 temperature 0 会直接拒绝，避免伪造解码条件。`resume: true` 会保留已有成功记录并只补缺失题目；`run_manifest.json` 会比对数据、题目、人设、模型与解码配置，配置改变时拒绝续跑。`max_calls` 是累计请求硬上限，成功和失败请求都计入，续跑不会重置额度。

部分响应即使请求禁用 thinking 仍可能包含 `<think>` 并耗尽输出上限。smoke 配置因此保留原始响应、将解析失败落盘，并使用较大的输出上限；正式实验前仍需锁定模型行为与预算。

每次 `pilot` 只运行一个“人设 × 状态 × 解码 × 模型”单元。完整设计需要由外层实验清单重复运行，并使用不同输出目录；不要在正式生成后修改预先锁定的关卡阈值。

## 已锁定的 100 人 pilot

当前有效配置 `configs/pilot_frcsub_minimax_m3_v1.yaml` 固定 `MiniMax-M3`、实际 temperature 1、无人设／属性人设、先答后解释／错误示例两种提示，共四个单元。每单元 100 人 × 20 题，目标共 8000 次学生请求；两次合成题预检和全部失败请求也计入 8800 次全局硬上限。详细设计与提示文本见 [`docs/pilot_preregistration_minimax_m3.md`](docs/pilot_preregistration_minimax_m3.md)。

旧 `gpt-4o-mini-2024-07-18`、temperature 0 方案已在任何正式请求发生前标记为 `superseded_before_execution`，仅保留历史，不能再执行。MiniMax 请求体不发送 temperature，避免把未生效的请求值误记为实际解码条件。

```bash
# 只校验设计：不读密钥、不读真实数据、不调用模型
PYTHONPATH=src python scripts/run_locked_pilot.py \
  --config configs/pilot_frcsub_minimax_m3_v1.yaml

# 只有显式 --execute 才会执行预检和付费学生生成
PYTHONPATH=src python scripts/run_locked_pilot.py \
  --config configs/pilot_frcsub_minimax_m3_v1.yaml --execute
```

预注册锁定时只完成离线测试与 dry run，尚未执行正式预检或正式生成。MiniMax-M3 是提供方模型名而非不可变快照；运行账本保留实际 provider model 和调用日期，但不声称后端权重绝对不变。

正式执行器最多并发处理 4 名虚拟学生；每名学生内部仍按既定拉丁方顺序逐题请求。所有并发任务共享同一调用计数和输出锁，失败与在途请求都写入账本，不能突破单元或全局上限。首个单元已有 870 条请求记录后曾短暂使用 8 学生并发，触发提供方 429 限流后降为 4；这些调度调整只改变执行效率，不改变模型、提示、人设、题目顺序、样本量、温度、预算或配置 SHA。四单元的真人 split-half 噪声界值完全相同，执行器会复用经作答矩阵 SHA、重复次数和随机种子校验的离线缓存，避免重复计算；缓存不包含 LLM 作答，也不改变指标。

完成运行后可按 [`docs/analysis.md`](docs/analysis.md) 执行离线审计，获得截断敏感性、DINA／G-DINA 同切分对照、SRMSR、逐属性后验分类置信度、留出对数似然差的配对 bootstrap 区间、题目对 Holm 检验、固定 Q 矩阵聚合敏感性、重复回读天花板和描述性关卡参照。正式审计必须用 `--dimension-b-model` 显式指定预注册模型；省略时只作探索性留出规则选择。另有独立命令执行重复切分与 item_008 剔除敏感性；真实输出均保留在本地。DINA 审计默认使用 10 组初值，DINA v2 配置指定 30 组初值和 300 次迭代上限。小样本审计不会自动给出正式推断或关卡通过结论。

## 方法边界

- Python 中的 DINA EM、logit-link G-DINA EM 与 2PL MML 是透明、可测试的轻量基线实现；G-DINA 使用弱 ridge 保证数值稳定，并记录收敛状态。SRMSR、题目对检验与配对 bootstrap 支持单次及重复切分的 pilot 诊断；recurrence 只表示切分稳定性，不是新的确认性 p 值。DINA 保留全部初值的训练似然与收敛记录。估计器验证以多初值、参数恢复和已发表参数对照为主，见 [`docs/estimator_validation.md`](docs/estimator_validation.md)；R `scripts/r_cdm_crosscheck.R` 仅为可选外部交叉检查，当前未执行。本仓库尚未实现混合效应 logistic 回归。
- 四分相关使用 0.5 连续性校正的成对最大似然估计，并限制在 ±0.95，适合小样本试点；正式分析需固定平滑规则并做敏感性分析。
- `noise_delta_95` 只是等效判定的一部分。正式“三档判定”还需要预先规定的 bootstrap CI、重复切分和多重比较方案，不能只看一个点估计。
- CDM 指标只作辅证，避免用 CDM 指标循环证明 CDM 生成基线优越。
- 完整作答模式 TV 距离在稀疏样本下只作描述。
- 有状态条件只保留该学生自己的历史；RQ3 的错配历史、干扰项 JS 散度与错误概念一致性将在有合格数据后实现。
- MiniMax-M3 的版本锁定只能精确到提供方模型名与调用记录，不能等同于固定权重快照；开源模型本地推理留待正式全实验。

## 仓库结构

```text
src/student_fidelity/
├── data.py                  # 数据校验、切分、合成 smoke-test 数据
├── tetrachoric.py           # 四分相关及最近 PSD 投影
├── metrics.py               # 保真度、噪声与可控性指标
├── baselines/               # Bernoulli / copula / 2PL / DINA / G-DINA
├── simulation/              # 人设、模型接口、逐题运行器
├── model_comparison.py      # 同切分 CDM 留出拟合对照
├── item_fit.py              # 后验模拟题目对检验与 Holm 校正
├── q_sensitivity.py         # 预先固定的属性聚合敏感性
├── repeated_splits.py       # 重复学生切分与局部失拟合 recurrence
├── item_sensitivity.py      # 保留全部题目的主分析＋剔除敏感性
├── parameter_recovery.py    # 参数恢复、SRMSR 与题目对拒绝率模拟
├── experiment.py            # 单个 pilot 单元编排
└── cli.py
```

研究原始计划见 [`docs/research_plan.md`](docs/research_plan.md)。

## 复现与数据安全

- 固定随机种子并保留每次运行的配置、模型版本、调用日期、温度、题目顺序和原始输出。
- 真实学生数据、API 密钥、运行产物均被 `.gitignore` 排除。
- 上传或公开数据前单独核对原始数据许可；本仓库不授予任何外部数据集的使用权。
