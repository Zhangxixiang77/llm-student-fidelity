# LLM Student Fidelity

用认知诊断模型检验 **LLM 模拟学生是否“答对了题，却不是同一个学生”**。本仓库把保真度与可控性分开评估：题目通过率吻合只是起点，题目间依赖结构、属性掌握结构和隐含 slip/guess 才是核心证据。

## 当前状态

这是一个可运行的 FrcSub pilot 方法骨架，不是正式实验结果。首版包含：

- 统一的学生 × 题目二元作答与 Q 矩阵接口，严格按学生切分拟合半/留出半；
- 独立 Bernoulli、高斯 copula、2PL IRT、DINA 四类生成基线；
- 题目通过率相关/RMSE、总分 Wasserstein 距离、四分相关矩阵 RMSE、完整作答模式 TV 距离；
- 真人 split-half 噪声分布与 95% 界值、生成条件 bootstrap CI 函数；
- 属性/能力/无人设，逐题无状态/有状态作答，题目位置循环拉丁方平衡；
- 属性与模式回读、DINA 可达天花板、隐含 slip/guess 与能力泄漏率；
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
├── raw_simulation.jsonl      # 每名虚拟学生每题的最终答案、演算与原始输出
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

MiniMax 的小样本配置见 `configs/pilot.minimax.yaml`。它使用 `MiniMax-M3`，从环境变量 `MINIMAX_API_KEY` 读取密钥，不发送 `temperature`，并把实际固定值 1.0 记录在元数据中；若请求 temperature 0 会直接拒绝，避免伪造解码条件。`resume: true` 会保留已有成功记录并只补缺失题目，只能用于同一数据、人设、题目顺序和模型配置的续跑。

部分响应即使请求禁用 thinking 仍可能包含 `<think>` 并耗尽输出上限。smoke 配置因此保留原始响应、将解析失败落盘，并使用较大的输出上限；正式实验前仍需锁定模型行为与预算。

每次 `pilot` 只运行一个“人设 × 状态 × 解码 × 模型”单元。完整设计需要由外层实验清单重复运行，并使用不同输出目录；不要在正式生成后修改预先锁定的关卡阈值。

## 方法边界

- Python 中的 DINA EM 与 2PL MML 是透明、可测试的轻量基线实现。正式论文应用 R 的 GDINA/CDM 复核参数、拟合和分类准确性；本仓库尚未实现 G-DINA、SRMSR、题目对检验或混合效应 logistic 回归。
- 四分相关使用 0.5 连续性校正的成对最大似然估计，并限制在 ±0.95，适合小样本试点；正式分析需固定平滑规则并做敏感性分析。
- `noise_delta_95` 只是等效判定的一部分。正式“三档判定”还需要预先规定的 bootstrap CI、重复切分和多重比较方案，不能只看一个点估计。
- CDM 指标只作辅证，避免用 CDM 指标循环证明 CDM 生成基线优越。
- 完整作答模式 TV 距离在稀疏样本下只作描述。
- 有状态条件只保留该学生自己的历史；RQ3 的错配历史、干扰项 JS 散度与错误概念一致性将在有合格数据后实现。
- 开源模型本地推理、闭源模型的版本锁定与批量容错需在确定 pilot 模型后补充。

## 仓库结构

```text
src/student_fidelity/
├── data.py                  # 数据校验、切分、合成 smoke-test 数据
├── tetrachoric.py           # 四分相关及最近 PSD 投影
├── metrics.py               # 保真度、噪声与可控性指标
├── baselines/               # Bernoulli / copula / 2PL / DINA
├── simulation/              # 人设、模型接口、逐题运行器
├── experiment.py            # 单个 pilot 单元编排
└── cli.py
```

研究原始计划见 [`docs/research_plan.md`](docs/research_plan.md)。

## 复现与数据安全

- 固定随机种子并保留每次运行的配置、模型版本、调用日期、温度、题目顺序和原始输出。
- 真实学生数据、API 密钥、运行产物均被 `.gitignore` 排除。
- 上传或公开数据前单独核对原始数据许可；本仓库不授予任何外部数据集的使用权。
