# 离线审计与 pilot 关卡

## 运行审计

完成一个 pilot 单元后，可以在不调用模型的情况下运行：

```bash
PYTHONPATH=src python scripts/audit_run.py \
  --dataset data/processed/frcsub \
  --run-dir runs/minimax_m3_attribute_smoke_v2 \
  --ceiling-repeats 100
```

输出 `audit.json`，包括请求数、失败数、finish reason、token usage、主要指标、DINA 拟合诊断、重复回读天花板和描述性关卡参照。真实数据与运行输出仍由 `.gitignore` 排除。

## `length` 敏感性

主分析保留具有完整、可解析 `final_answer` 的作答，同时单独报告 `finish_reason=length`。敏感性分析采用保守的完整学生剔除规则：只要某名学生有一条 `length` 记录，就剔除该学生的全部作答，不对单题进行插补。因此它同时反映截断处理和学生构成变化，样本很小时只能作描述。

若答案 JSON 本身不完整或无法评分，该请求仍是失败记录，不能进入主分析。正式实验前应预注册截断规则，不能用看到结果后的有利口径替换主规则。

## DINA 诊断的含义

- `holdout_srmsr` 比较留出真人作答的逐题 Pearson 相关与拟合 DINA 在估计掌握模式先验下的模型隐含相关；0.05 是研究计划中的暂定参考，不是普遍适用的硬阈值。
- `attribute_classification_confidence` 是后验 MAP 分类置信度，即模型内部自一致性，不是相对外部金标准的分类准确性。
- `readback_ceiling` 对同一批指定掌握模式重复生成并回读，报告逐属性与模式层可达天花板；重复次数写入结果。
- 能力泄漏必须与真人拟合 guess 及其拟合质量一起解释。若真人 DINA 拟合不足，单独比较泄漏率不能通过方法关卡。

当前实现尚未完成 G-DINA、题目对检验和外部分类准确性验证，因此 `audit.json` 不会自动宣布 pilot 关卡通过。生成学生数很小时不运行 bootstrap 三档判定；正式推断仍需预先规定的样本量、重复切分和置信区间方案。
