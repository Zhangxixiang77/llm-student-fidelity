# Python 认知诊断估计器验证

估计方法复核的核心是排除 Python 实现错误，而不是必须使用某一种外部软件。本仓库采用三项互补证据：真实数据多初值稳定性、正确设定模型的参数恢复模拟、已发表全量参数对照。R `GDINA` 脚本只保留为可选外部交叉检查，不是验证成立的前提。

## 1. 真实数据多初值

DINA 的第 0 组初值固定为均匀掌握模式先验和 slip＝guess＝0.1；后续初值分别随机化掌握模式先验、slip 和 guess。每组独立运行 EM，以最终训练边际对数似然选优，并保留所有初值的似然、迭代次数和收敛状态。真实 FrcSub 重复切分使用 30 组初值：

```bash
PYTHONPATH=src python scripts/repeated_split_diagnostics.py \
  --dataset data/processed/frcsub \
  --output runs/frcsub_repeated_splits_dina30/repeated_split_diagnostics.json \
  --splits 20 --dina-starts 30 --dina-max-iter 300 --gdina-max-iter 300
```

比较稳定的最优训练似然、留出对数似然、SRMSR 和收敛记录，而不只比较某个点估计。多初值不能修复错误的 EM 更新，因此必须与下面两项验证组合使用。

修复版结果：全量 30 组初值全部收敛，最优训练对数似然仅在 −4402.306274 到 −4402.302920 间变化，范围 0.003354。20 个重复切分中，每次 DINA 最优解均收敛且 30／30 初值均收敛；相对确定性初值的训练似然增益中位数仅 0.000225，最大 0.725857。20／20 切分的 DINA 留出预测优于 G-DINA，G-DINA−DINA 平均差为 −0.156313（逐切分范围 −0.450541 到 −0.003275），其中 10 个配对 bootstrap 区间完全低于零，另 10 个包含零。两种模型均未达到 SRMSR ≤ 0.05。旧版基于错误 DINA 的“20／20 暂选 G-DINA”结论作废；这项新证据是正式 pilot 前重新决定维度 B 模型的依据，而不是事后修改锁定关卡的理由。

## 2. 参数恢复、SRMSR 与假阳性率

`src/student_fidelity/parameter_recovery.py` 实现正确设定的参数自举：

1. 在全量真人 FrcSub 上分别估计 DINA 与 G-DINA，作为模拟的生成参数；
2. 按估计的掌握模式先验和题目成功概率，在 N＝268、536 下生成二元作答；
3. 用同一实现重新拟合正确模型；
4. DINA 比较逐题 guess／slip 的 bias、MAE、RMSE 和相关，另报告掌握模式先验误差；
5. G-DINA 比较稳定且可解释的 `p_correct_`，同时给未加权和生成先验加权误差，不直接比较有 ridge 的 beta；
6. 每次重复计算 SRMSR，以及 transformed-correlation 与 log-OR 题目对检验在未校正、test-level Holm 和 item-level Holm 后的拒绝率；
7. 保存逐次收敛状态，并按重复估计 Monte Carlo 标准误和区间。题目对拒绝率先在每次重复内汇总，再跨重复估计不确定性，不把同一数据集中的 190 个题目对当作独立重复。

正式本地运行：

```bash
PYTHONPATH=src python scripts/parameter_recovery.py \
  --data-dir data/processed/frcsub \
  --output-dir runs/parameter_recovery_v1 \
  --models dina gdina --sample-sizes 268 536 \
  --replications 100 --dina-starts 10 \
  --generating-dina-starts 30 --itemfit-simulations 100000 --resume
```

生成模型使用 30 组 DINA 初值；为控制 Monte Carlo 成本，每次重拟合使用 10 组初值并单独审计收敛。每次重复写入独立检查点，`--resume` 只补缺失重复。100 次重复仍不能精确证明名义 0.05 水平；结果必须连同 Monte Carlo 不确定性报告。正确设定模拟中的低 SRMSR 和适当拒绝率只支持实现与检验校准，不证明真人数据上的模型设定正确。

### 100 次／条件的结果

四个条件各 100 次，共 400 个完整且不重复的本地检查点。四种条件的最优解收敛率均为 100%；DINA 在两种样本量下均有 99% 的重复达到全部 10 个初值收敛，未收敛初值没有被静默当作最优解。下表括号内是均值跨重复的 Monte Carlo 标准误；G-DINA 的 MAE 针对全部掌握模式 × 题目的 `p_correct_`。

| 模型 | N | 参数 MAE（MCSE） | SRMSR 均值（MCSE） |
| --- | ---: | --- | --- |
| DINA | 268 | guess 0.0253（0.0006）；slip 0.0231（0.0005） | 0.0371（0.0004） |
| DINA | 536 | guess 0.0183（0.0004）；slip 0.0163（0.0003） | 0.0255（0.0002） |
| G-DINA | 268 | `p_correct_` 0.0742（0.0014） | 0.0326（0.0003） |
| G-DINA | 536 | `p_correct_` 0.0497（0.0012） | 0.0228（0.0002） |

DINA 的 guess／slip 相关在 N＝268 时分别为 0.952／0.937，在 N＝536 时升至 0.971／0.968；两类参数误差随样本量下降。G-DINA 的 `p_correct_` RMSE 从 0.1793 降至 0.1327，生成先验加权 RMSE 从 0.1135 降至 0.0765，相关从 0.888 升至 0.936。它的高维概率估计明显比 DINA 逐题参数更 noisy，不能把拟合检验校准良好解释为所有 G-DINA 参数都恢复精确。掌握模式先验也仍难恢复：DINA 的先验总变差均值为 0.272／0.252，G-DINA 为 0.400／0.296。

题目对检验的结果按“每次重复是否至少拒绝一对”汇总家族错误率，并给 100 次重复的 Wilson 区间：

| 模型 | N | 相关残差：test-level Holm［95% 区间］ | log-OR：test-level Holm［95% 区间］ |
| --- | ---: | --- | --- |
| DINA | 268 | 0.01［0.002，0.054］ | 0.06［0.028，0.125］ |
| DINA | 536 | 0.00［0.000，0.037］ | 0.01［0.002，0.054］ |
| G-DINA | 268 | 0.00［0.000，0.037］ | 0.05［0.022，0.112］ |
| G-DINA | 536 | 0.00［0.000，0.037］ | 0.02［0.006，0.070］ |

未校正逐题目对拒绝率均低于名义 0.05：相关残差为 0.0046–0.0095，log-OR 为 0.0069–0.0126，表现为保守而非过拒。Holm 后也未观察到失控的家族错误率；有限重复的区间仍较宽，尤其不能将 0／100 解释成真实错误率为零。完整逐次结果、item-level Holm、稀疏单元格诊断和区间保留在本地 `summary_*.json`。

## 3. 已发表全量参数对照

`scripts/published_dina_crosscheck.py` 使用全量 536 人、20 题、8 属性数据和 30 组初值拟合 DINA，与 Sorrel 等（2023）Table 1 的 GDINA MMLE-EM 列逐题比较。该文使用同一个 GDINA `frac20` 数据与 Q 矩阵，因此比只含 15 题子集或不同 Q 矩阵的表更适合作为数值对照。

```bash
PYTHONPATH=src python scripts/published_dina_crosscheck.py \
  --data-dir data/processed/frcsub \
  --output-dir runs/published_dina_crosscheck_v1 \
  --dina-starts 30 --dina-max-iter 300
```

修复版实现的外部对照结果：

- 训练边际对数似然 −4402.302920；AIC 9394.605840，公开值 9394.606；BIC 10658.425418，公开值 10658.430；
- 30／30 初值收敛；
- guess 相对公开三位小数表的 MAE 0.000289、最大绝对差 0.000501；
- slip 的 MAE 0.000258、最大绝对差 0.000518；
- 两类参数与公开表的相关系数均大于 0.99999。

文献：Sorrel, M. A., Escudero, S., Nájera, P., Kreitchmann, R. S., & Vázquez-Lira, R. (2023). [Exploring Approaches for Estimating Parameters in Cognitive Diagnosis Models with Small Sample Sizes](https://doi.org/10.3390/psych5020023). *Psych, 5*, 336–349，Table 1。

这项对照曾暴露旧版 DINA 将逐学生非掌握概率误写为 `N − expected_mastery` 的实现错误；正确公式为 `1 − expected_mastery`。错误会把 guess 推向题目总体正确率。修复后增加了第一步 EM 的解析回归测试，旧本地 DINA 数值全部作废并重跑，不能与新结果混合引用。G-DINA 的逐题加权 logistic 更新不使用这段代码。

## 结果与数据边界

`runs/`、真实作答、逐次模拟记录和生成参数均保持本地，不上传仓库。仓库只提供透明实现、测试、公开文献常数和运行方法。参数恢复通过不等于 FrcSub 的 DINA／G-DINA 拟合通过，也不等于 LLM 模拟学生的保真度关卡通过。
