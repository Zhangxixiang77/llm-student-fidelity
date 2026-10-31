# 离线审计与 pilot 关卡

## 运行审计

完成一个 pilot 单元后，可以在不调用模型的情况下运行：

```bash
PYTHONPATH=src python scripts/audit_run.py \
  --dataset data/processed/frcsub \
  --run-dir runs/minimax_m3_attribute_smoke_v2 \
  --ceiling-repeats 100 \
  --comparison-bootstraps 1000
```

输出 `audit.json`，包括请求数、失败数、finish reason、token usage、主要指标、DINA／G-DINA 同切分拟合对照、两种模型的重复回读天花板和描述性关卡参照。真实数据与运行输出仍由 `.gitignore` 排除。

## `length` 敏感性

主分析保留具有完整、可解析 `final_answer` 的作答，同时单独报告 `finish_reason=length`。敏感性分析采用保守的完整学生剔除规则：只要某名学生有一条 `length` 记录，就剔除该学生的全部作答，不对单题进行插补。因此它同时反映截断处理和学生构成变化，样本很小时只能作描述。

若答案 JSON 本身不完整或无法评分，该请求仍是失败记录，不能进入主分析。正式实验前应预注册截断规则，不能用看到结果后的有利口径替换主规则。

## DINA／G-DINA 同切分对照

审计在同一个真人拟合半上估计 DINA 与 logit-link G-DINA，并在同一个留出半上比较逐学生边际对数似然。`gdina_minus_dina_mean_holdout_log_likelihood` 为正表示 G-DINA 的留出预测更好；`paired_student_bootstrap_95_ci` 通过对学生成对重采样反映这一单次切分的不确定性。区间下界大于零时，`provisional_dimension_b_model` 暂选 G-DINA，否则暂选 DINA。

这只是研究计划中模型选择规则的 pilot 实现，不等于关卡 1 通过。G-DINA 使用饱和题目反应函数、弱 ridge 和确定性的 DINA 式初始化；审计记录迭代次数、收敛状态、正则化强度和名义参数量。正式分析仍需用 R 的 GDINA／CDM 复核，并完成重复切分、题目对检验和外部分类准确性验证。

当暂选 G-DINA 时，实际回读也使用 G-DINA。`gdina_mean_residual` 是观察正确率减指定掌握模式下的模型预测正确率；`gdina_mean_master_shortfall` 在 DINA 理想掌握题上反向计算该残差，`gdina_mean_nonmaster_excess` 在非掌握题上正向计算。后两者是模型残差，不能与原始 DINA slip／guess 或关卡 3 的 0.30 界值直接混用。

## DINA 诊断的含义

- `holdout_srmsr` 比较留出真人作答的逐题 Pearson 相关与拟合 DINA 在估计掌握模式先验下的模型隐含相关；0.05 是研究计划中的暂定参考，不是普遍适用的硬阈值。
- `attribute_classification_confidence` 是逐属性边际后验分类置信度，即模型内部自一致性，不是相对外部金标准的分类准确性。
- `readback_ceiling` 对同一批指定掌握模式重复生成并回读，报告逐属性与模式层可达天花板；重复次数写入结果。
- 能力泄漏必须与真人拟合 guess 及其拟合质量一起解释。若真人 DINA 拟合不足，单独比较泄漏率不能通过方法关卡。

当前实现尚未完成题目对检验、外部分类准确性验证和重复切分，因此 `audit.json` 不会自动宣布 pilot 关卡通过。生成学生数很小时不运行 bootstrap 三档判定；CDM 对照中的配对 bootstrap 只描述真人留出拟合差，不是 LLM 指标的正式推断。正式分析仍需预先规定的样本量、重复切分和置信区间方案。
