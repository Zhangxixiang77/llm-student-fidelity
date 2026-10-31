# 离线审计与 pilot 关卡

## 运行审计

完成一个 pilot 单元后，可以在不调用模型的情况下运行：

```bash
PYTHONPATH=src python scripts/audit_run.py \
  --dataset data/processed/frcsub \
  --run-dir runs/minimax_m3_attribute_smoke_v2 \
  --ceiling-repeats 100 \
  --comparison-bootstraps 1000 \
  --itemfit-simulations 100000
```

输出 `audit.json`，包括请求数、失败数、finish reason、token usage、主要指标、DINA／G-DINA 同切分拟合对照、留出题目对检验、固定 Q 矩阵敏感性、两种模型的重复回读天花板和描述性关卡参照。真实数据与运行输出仍由 `.gitignore` 排除。

## `length` 敏感性

主分析保留具有完整、可解析 `final_answer` 的作答，同时单独报告 `finish_reason=length`。敏感性分析采用保守的完整学生剔除规则：只要某名学生有一条 `length` 记录，就剔除该学生的全部作答，不对单题进行插补。因此它同时反映截断处理和学生构成变化，样本很小时只能作描述。

若答案 JSON 本身不完整或无法评分，该请求仍是失败记录，不能进入主分析。正式实验前应预注册截断规则，不能用看到结果后的有利口径替换主规则。

## DINA／G-DINA 同切分对照

审计在同一个真人拟合半上估计 DINA 与 logit-link G-DINA，并在同一个留出半上比较逐学生边际对数似然。`gdina_minus_dina_mean_holdout_log_likelihood` 为正表示 G-DINA 的留出预测更好；`paired_student_bootstrap_95_ci` 通过对学生成对重采样反映这一单次切分的不确定性。区间下界大于零时，`provisional_dimension_b_model` 暂选 G-DINA，否则暂选 DINA。

这只是研究计划中模型选择规则的 pilot 实现，不等于关卡 1 通过。G-DINA 使用饱和题目反应函数、弱 ridge 和确定性的 DINA 式初始化；审计记录迭代次数、收敛状态、正则化强度和名义参数量。正式分析仍需用 R 的 GDINA／CDM 复核，并完成重复切分、题目对检验和外部分类准确性验证。

当暂选 G-DINA 时，实际回读也使用 G-DINA。`gdina_mean_residual` 是观察正确率减指定掌握模式下的模型预测正确率；`gdina_mean_master_shortfall` 在 DINA 理想掌握题上反向计算该残差，`gdina_mean_nonmaster_excess` 在非掌握题上正向计算。后两者是模型残差，不能与原始 DINA slip／guess 或关卡 3 的 0.30 界值直接混用。

## 留出题目对拟合检验

`item_pair_fit` 分别对拟合半上估计的 DINA 与 G-DINA 做诊断，观察数据始终是同一个真人留出半，不用留出数据重新估计参数。实现参照 Chen、de la Torre、Zhang（2013）及 R `GDINA::itemfit`：先从留出学生的平均逐模式后验中重复抽取掌握模式，再按模型逐题概率生成期望作答；默认模拟人数为 `max(100000, 10*N)`，也可用 `--itemfit-simulations` 显式固定。

每个题目对报告两类统计：

- **transformed correlation**：观察与期望 Pearson 相关分别做 Fisher-z 变换后取差，标准误为 `sqrt(1/(N-3))`；
- **log-odds-ratio**：观察与期望 2×2 表的 log OR 取差，标准误来自期望表四个单元格倒数和，并乘后验重复模拟倍数。

两种统计分别进行正态近似双侧检验。test-level Holm 校正的检验族为全部 `J(J-1)/2` 个题目对；item-level Holm 校正分别在每道题的 `J-1` 个搭档内进行，并报告最大 z 值及最差搭档。log OR 遇到零单元格时仅把该单元格计数替换为 0.5，同时明确记录。期望单元格按观察样本量缩放后小于 5 的题目对会被标记；该标记用于限制正态近似解释，不是另一项失拟合检验，也不会静默删除题目对。

显著题目对只能说明模型未重现的局部依赖，不能单独区分反应函数失配、Q 矩阵失配、题目内容导致的局部依赖或单次切分噪声。正式结论仍需重复切分和 R GDINA／CDM 复核。

## 预先固定的 Q 矩阵／属性聚合敏感性

对具有标准 FrcSub 八属性名称的数据，审计固定评估以下候选，且都在看到题目对结果之前写入同一代码路径：

1. `original_8`：原始八属性矩阵，始终是主分析；
2. `mixed_number_7`：合并整数转分数与分离整数／分数部分；
3. `simplification_7`：合并相减前化简与结果约分；
4. `borrowing_7`：合并整数部分借位与竖式借位；
5. `combined_5`：同时实施上述三组内容层合并。

合并只对组内 Q 列做逐题逻辑 OR，不改单个题目 Q 条目，不搜索替代矩阵。每个候选在同一拟合半重新估计 G-DINA，并在同一留出半计算平均逐学生对数似然、相对原矩阵的差及学生配对 bootstrap 区间，同时报告 SRMSR、名义参数量和收敛状态。该分析不自动选择“最优 Q”，也不对多个候选做事后确认性筛选；即使某候选预测更好，也必须结合内容有效性、题目对残差和重复切分，不能把统计改善直接解释为 Q 矩阵得到验证。

## DINA 诊断的含义

- `holdout_srmsr` 比较留出真人作答的逐题 Pearson 相关与拟合 DINA 在估计掌握模式先验下的模型隐含相关；0.05 是研究计划中的暂定参考，不是普遍适用的硬阈值。
- `attribute_classification_confidence` 是逐属性边际后验分类置信度，即模型内部自一致性，不是相对外部金标准的分类准确性。
- `readback_ceiling` 对同一批指定掌握模式重复生成并回读，报告逐属性与模式层可达天花板；重复次数写入结果。
- 能力泄漏必须与真人拟合 guess 及其拟合质量一起解释。若真人 DINA 拟合不足，单独比较泄漏率不能通过方法关卡。

当前实现尚未完成外部分类准确性验证和重复切分，因此 `audit.json` 不会自动宣布 pilot 关卡通过。生成学生数很小时不运行 bootstrap 三档判定；CDM 对照与 Q 敏感性中的配对 bootstrap 只描述真人留出拟合差，不是 LLM 指标的正式推断。正式分析仍需预先规定的样本量、重复切分和置信区间方案。
