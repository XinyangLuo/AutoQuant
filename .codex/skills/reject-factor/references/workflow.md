# Reject Factor Workflow

本 workflow 用于正式拒绝并归档失败因子。它是独立 skill：只做 reject、清理、归档和 KB 检查，不做 PDF hypothesis 或因子迭代。

## 1. 候选发现

如果用户没有提供 `factor_id`，先发现候选并展示编号菜单：

1. 运行 admission status：

```bash
conda activate AutoQuant && python -m backtest.factor.admission status
```

2. 遍历 `alphas/exp/agent/` 下的 `f_auto_*` 目录。
3. 遍历 `results/` 下的 `f_auto_*` 目录，排除 `results/candidates/` 和 `results/rejected/`。
4. 合并候选，按状态优先级排序：pending/failed、仅有代码、仅有 results、already rejected。

菜单格式：

```text
可 reject 的因子：
1. f_auto_20260612_001 | pending | code yes | results yes | KB fail record yes
2. f_auto_20260610_003 | code only | code yes | results no | KB fail record no
```

不要要求用户手写 factor_id；用户只需选择编号。

## 2. 操作确认

选择因子后，展示确认摘要并等待用户明确确认：

```text
将执行：
- admission reject: f_auto_xxx
- code: 删除 alphas/exp/agent/f_auto_xxx/   # 或保留
- results: 移动 results/f_auto_xxx/ 到 results/rejected/f_auto_xxx/
- run traces: 移动匹配的 results/*f_auto_xxx*run*/ 到 results/rejected/
- KB: 检查 failed_attempts.jsonl 和 anti_patterns.json
```

用户明确确认前，不得运行 DB reject、`rm`、`mv` 或其他 destructive 操作。

## 3. Reject 与清理

执行 admission reject：

```bash
conda activate AutoQuant && python -m backtest.factor.admission reject <factor_id> \
  --notes "<concise English notes>"
```

代码目录：

- 默认删除 `alphas/exp/agent/<factor_id>/`。
- 如果用户要求保留代码，则跳过删除并在总结中标注。

结果归档：

```bash
mkdir -p results/rejected
mv results/<factor_id>/ results/rejected/<factor_id>/
```

如果存在匹配 run trace 目录，也移动到 `results/rejected/`。不要覆盖已有归档；如目标已存在，追加时间戳后缀。

## 4. KB 检查

检查：

```bash
grep -q '<factor_id>' agents/knowledge_base/failed_attempts.jsonl
grep -q '<factor_id>' agents/knowledge_base/anti_patterns.json
```

- `failed_attempts.jsonl` 缺失时，提示该失败没有沉淀到 KB。
- `anti_patterns.json` 缺失不一定是错误；只有 RC 发现新反模式时才应存在。

## 5. 最终输出

用表格汇总：

| 操作 | 状态 |
|------|------|
| admission reject | done / skipped / failed |
| work DB 清理 | done / failed |
| registry 标记 rejected | done / failed |
| 临时代码 | deleted / kept / missing |
| results 归档 | archived / missing / failed |
| KB failed_attempts | present / missing |
| KB anti_patterns | present / not required / missing |

如果 reject 失败是因为因子已 admitted，停止并提示用户需要先走 unadmit 流程；不要强行清理 admitted 因子。
