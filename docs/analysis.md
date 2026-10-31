# 离线审计与 pilot 关卡

## 运行审计

完成一个 pilot 单元后，可以在不调用模型的情况下运行：

```bash
PYTHONPATH=src python scripts/audit_run.py \
  --dataset data/processed/frcsub \
  --run-dir runs/minimax_m3_attribute_smoke_v2 \
  --ceiling-repeats 100 \
  --comparison-bootstraps 1000 \
  --itemfit-simulations 100000 \
  --dina-starts 10
```

输出 `audit.json`，包括请求数、失败数、finish reason、token usage、主要指标、DINA／G-DINA 同切分拟合对照、留出题目对检验、固定 Q 矩阵敏感性、两种模型的重复回读天花板和描述性关卡参照。真实数据与运行输出仍由 `.gitignore` 排除。

## `length` 敏感性

主分析保留具有完整、可解析 `final_answer` 的作答，同时单独报告 `finish_reason=length`。敏感性分析采用保守的完整学生剔除规则：只要某名学生有一条 `length` 记录，就剔除该学生的全部作答，不对单题进行插补。因此它同时反映截断处理和学生构成变化，样本很小时只能作描述。

若答案 JSON 本身不完整或无法评分，该请求仍是失败记录，不能进入主分析。正式实验前应预注册截断规则，不能用看到结果后的有利口径替换主规则。

## DINA／G-DINA 同切分对照

审计在同一个真人拟合半上估计 DINA 与 logit-link G-DINA，并在同一个留出半上比较逐学生边际对数似然。`gdina_minus_dina_mean_holdout_log_likelihood` 为正表示 G-DINA 的留出预测更好；`paired_student_bootstrap_95_ci` 通过对学生成对重采样反映这一单次切分的不确定性。区间下界大于零时，`provisional_dimension_b_model` 暂选 G-DINA，否则暂选 DINA。

这只是研究计划中模型选择规则的 pilot 实现，不等于关卡 1 通过。G-DINA 使用饱和题目反应函数、弱 ridge 和确定性的 DINA 式初始化；审计记录迭代次数、收敛状态、正则化强度和名义参数量。DINA 默认使用 1 组原确定性初值和 9 组可复现随机初值；随机初值从 Dirichlet 掌握模式先验及独立 slip/guess 区间抽取。每组独立运行 EM，按最终训练边际对数似然选优，完整保留各组似然、迭代次数和收敛状态，不因未收敛而静默丢弃。正式分析仍需用 R 的 GDINA／CDM 复核并完成外部分类准确性验证。重复切分与题目对检验由下方独立命令执行。

当暂选 G-DINA 时，实际回读也使用 G-DINA。`gdina_mean_residual` 是观察正确率减指定掌握模式下的模型预测正确率；`gdina_mean_master_shortfall` 在 DINA 理想掌握题上反向计算该残差，`gdina_mean_nonmaster_excess` 在非掌握题上正向计算。关卡 3 已在 100 人 pilot 前锁定为后者在全部 20 题上的未加权均值不超过 0.30；旧的 DINA guess＋0.30 参照不再使用。锁定规则见 `configs/pilot_gates.frcsub.yaml`，运行后不得根据结果调整。

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

## 重复学生切分

以下命令完全离线，按学生重复划分拟合半与留出半，每次重新估计 DINA／G-DINA，并检查模型选择、SRMSR、item_008 及 item_004–item_008 残差是否 recurring：

```bash
PYTHONPATH=src python scripts/repeated_split_diagnostics.py \
  --dataset data/processed/frcsub \
  --output runs/frcsub_repeated_splits/repeated_splits.json \
  --splits 20 \
  --seed 42 \
  --comparison-bootstraps 200 \
  --itemfit-simulations 100000 \
  --gdina-max-iter 300 \
  --dina-max-iter 300 \
  --dina-starts 10
```

`--gdina-max-iter 300` 与 `--dina-max-iter 300` 是计算收敛设置：较低的上限可能让部分切分尚未收敛，未收敛结果不应直接作实质解释。每次运行仍须检查两种模型的收敛状态和迭代次数。DINA 的 `dina_multistart_training_log_likelihood_gain` 衡量最优初值相对原确定性初值的训练似然增益；若随机初值只带来很小增益而留出模型差保持不变，才可说明主要结论不太像由单次初始化的局部最优造成。recurrence 描述对切分的稳定性，不是把相关切分当作独立样本生成新的确认性 p 值，也不会自动删题或改换维度 B 的模型。

## R GDINA 包独立复核（尚未执行）

`scripts/r_cdm_crosscheck.R` 使用 R `GDINA` 包分别估计 DINA 与 logit-link G-DINA，按完整掌握模式概率和掌握模式先验手动计算留出学生的边际对数似然。DINA 使用 `nstarts` 多初值；输出逐切分训练／留出指标和配对差汇总。该脚本不调用任何 LLM，也不把训练后验当作留出预测。

```bash
install.packages(c("GDINA", "jsonlite"))  # 在 R 交互环境中执行一次
Rscript scripts/r_cdm_crosscheck.R \
  --dataset data/processed/frcsub \
  --output runs/r_cdm_crosscheck \
  --splits 20 --seed 42 --dina-starts 10 --max-iter 300
```

默认 R 切分使用 R 自己的随机数流，不能宣称与 NumPy 切分完全相同。若提供 `--split-indices`，脚本可读取 `split_index`、`role` 及 `student_index` 或 `student_id`，从而复用外部导出的相同学生活动。当前开发环境无 R／Rscript，用户态安装又受网络解析限制，因此只完成了脚本与静态检查，**尚未完成 R 数值复核**。

## item_008 剔除敏感性

Köhn 等的 [A Two-Step Q-Matrix Estimation Method](https://pmc.ncbi.nlm.nih.gov/articles/PMC11560062/)（2024 年在线发表，2025 年卷期）在 FrcSub 内容审查中指出，item_008 可能还依赖原八属性之外的知识，例如“任何数减去自身为零”或加法逆元性质，并建议剔除该题以降低噪声。这为**敏感性分析**提供外部内容依据，但不是看到残差后静默删题的理由。

```bash
PYTHONPATH=src python scripts/item008_sensitivity.py \
  --dataset data/processed/frcsub \
  --output runs/frcsub_item008_sensitivity/item008_sensitivity.json \
  --exclude-items item_008 \
  --splits 20 \
  --seed 42 \
  --itemfit-simulations 100000 \
  --gdina-max-iter 300
```

每次切分在相同学生活动上分别拟合 full 与 reduced G-DINA。主分析始终为 `all_items_retained`；reduced 只移除完整题目列及其 Q 行，保留所有学生。由于两种分析的题目集不同，留出对数似然按每题报告，不直接用来选择题目集；SRMSR 与显著题目对数量也只是描述性敏感性诊断。即使 reduced 拟合改善，也不能据此宣称原八属性 Q 矩阵得到验证或关卡 1 自动通过。

## 属性层回读与锁定关卡

FrcSub 主 Q 矩阵始终为 `original_8`。模式层回读不进入正式分析；属性层只分析回读天花板至少为 0.80 的四个预定属性：`separate_whole_from_fraction`、`borrow_from_whole_part`、`subtract_numerators` 和 `reduce_to_simplest_form`。审计同时输出 `gdina_eligible_attribute_readback_agreement` 及逐属性结果，配置一致性由测试检查。

关卡 3 的指标、聚合方式、题目范围、0.30 界值与长度敏感性角色均锁定在 `configs/pilot_gates.frcsub.yaml`。当前配置只锁定规则，不表示 100 人 pilot 已运行或通过。MiniMax-M3 固定温度结果只用于 smoke，不进入正式实验。

## DINA 诊断的含义

- `holdout_srmsr` 比较留出真人作答的逐题 Pearson 相关与拟合 DINA 在估计掌握模式先验下的模型隐含相关；0.05 是研究计划中的暂定参考，不是普遍适用的硬阈值。
- `attribute_classification_confidence` 是逐属性边际后验分类置信度，即模型内部自一致性，不是相对外部金标准的分类准确性。
- `readback_ceiling` 对同一批指定掌握模式重复生成并回读，报告逐属性与模式层可达天花板；重复次数写入结果。
- 能力泄漏必须与真人拟合 guess 及其拟合质量一起解释。若真人 DINA 拟合不足，单独比较泄漏率不能通过方法关卡。

当前实现尚未完成外部分类准确性验证和 R GDINA／CDM 复核，因此 `audit.json` 不会自动宣布 pilot 关卡通过。生成学生数很小时不运行 bootstrap 三档判定；CDM 对照与 Q 敏感性中的配对 bootstrap 只描述真人留出拟合差，不是 LLM 指标的正式推断。正式分析仍需预先规定的样本量、重复切分和置信区间方案。
