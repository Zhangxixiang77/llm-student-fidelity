# Hugging Face 本地开放权重模型：候选与预检

## 当前状态

`configs/huggingface_candidates_v1.yaml` 是**候选预检配置，不是正式学生生成预注册**。它保留四单元、每模型 100 人 × 20 题、每模型 8000 次目标学生请求和 8800 次累计上限，作为后续锁定正式设计时的比较基线；当前脚本只允许解析元数据、显式下载权重和运行合成题预检，不能启动学生生成。

三个候选为：

| 候选 | Hugging Face 仓库 | 配置中要求的许可证 | 当前 revision |
| --- | --- | --- | --- |
| Qwen3 4B | `Qwen/Qwen3-4B-Instruct-2507` | Apache-2.0 | 待官方元数据解析 |
| Phi-4 mini | `microsoft/Phi-4-mini-instruct` | MIT | 待官方元数据解析 |
| Mistral Nemo | `mistralai/Mistral-Nemo-Instruct-2407` | Apache-2.0 | 待官方元数据解析 |

当前执行环境没有 NVIDIA GPU，约 4GB 内存，且 Hugging Face 元数据请求发生网络超时。因此尚未解析完整 commit SHA、下载权重或执行真实本地模型预检。配置中的 `revision: null` 是有意的安全阻断：只有模型名、分支名或短 SHA 均不能执行。不得手工编造版本号绕过。

## 安全边界

- 默认 dry run 不访问网络、不读取 `HF_TOKEN`、不读取 FrcSub、不导入深度学习框架，也不创建输出目录。
- 只有 `--resolve-metadata` 才访问 Hugging Face 模型元数据接口；该步骤不下载权重。
- 只有 `--download` 才下载权重，且必须先提供与候选配置 SHA 绑定的元数据 JSON。
- 只有 `--execute` 才加载本地模型并运行预检；生成阶段强制 `local_files_only: true` 和 `trust_remote_code: false`。
- 每个模型必须使用完整 40 位 Hugging Face commit SHA，元数据中的仓库与许可证必须与候选配置一致。
- `HF_TOKEN` 只在 gated 仓库确实需要时通过进程环境提供；不得写入 YAML、JSON、命令行、日志或 GitHub。
- `.cache/`、`runs/`、真实数据、预检原始输出和模型权重均不上传仓库。

## 在联网 GPU 机器上的执行顺序

安装本地推理的可选依赖：

```bash
python -m pip install -e '.[hf,dev]'
```

先检查设计，不产生网络或模型活动：

```bash
PYTHONPATH=src python scripts/run_huggingface_preflight.py \
  --config configs/huggingface_candidates_v1.yaml
```

解析官方元数据并固定三个 revision：

```bash
PYTHONPATH=src python scripts/run_huggingface_preflight.py \
  --config configs/huggingface_candidates_v1.yaml --resolve-metadata
```

结果写入本地 `runs/huggingface_preflight_v1/resolved_models.json`。若仓库要求授权，先在当前进程环境中设置 `HF_TOKEN`；令牌不会进入结果文件。

显式下载已固定版本的模型快照：

```bash
PYTHONPATH=src python scripts/run_huggingface_preflight.py \
  --config configs/huggingface_candidates_v1.yaml \
  --metadata runs/huggingface_preflight_v1/resolved_models.json --download
```

运行本地预检：

```bash
PYTHONPATH=src python scripts/run_huggingface_preflight.py \
  --config configs/huggingface_candidates_v1.yaml \
  --metadata runs/huggingface_preflight_v1/resolved_models.json --execute
```

三个模型顺序加载和释放，不需要同时占用三张 GPU。实际显存需求取决于权重精度、上下文、推理框架和并发；当前候选配置使用 CUDA、bfloat16、单学生顺序推理。正式运行前应在目标 GPU 上验证，不应把模型免费理解为计算资源无限。

## 预检内容与记账

每个模型用合成题 `Compute 1/2 + 1/4.` 分别运行 `answer_first` 和 `error_examples`，共 6 次目标请求；所有成功、解析失败、加载后推理失败和续跑请求合计不得超过 12 次。提示模板与 MiniMax pilot 共用同一个构造函数，避免后端切换时改变提示。

预检保存实际仓库、完整 revision、finish reason、输入／输出 token 数、生成种子、是否采样、原始输出和最终答案。技术通过要求为仓库与 revision 完全匹配、`finish_reason=stop` 且 `final_answer` 可解析；合成题数学正确性另行记录，不把一道题当作能力认证。已经成功且配置指纹一致的检查在续跑时跳过，不会重复请求。

预检完成后仍需审查结构化成功率、截断、数学作答和提示遵循，再决定保留或替换候选。之后必须另建正式配置，锁定模型 revision、依赖版本、硬件与精度、解码参数、输出上限、样本量、重复设计和调用预算；本候选配置不授权正式十二单元学生生成。
