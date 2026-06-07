# /reject-factor

清理并归档一个失败的因子实验。执行正式的 `reject` 流程：清理 work DB、标记 registry、删除临时代码目录，并提示检查 KB 记录。

## Usage

```text
/reject-factor f_auto_20260602_001
/reject-factor                                         # 无参数 → 列出 pending/rejected 因子供选择
/reject-factor f_auto_xxx --keep-code                  # 只清理 DB 和 registry，保留 alphas/exp/agent/ 代码
```

## Operating Rules

1. **调用 admission reject**：
   ```bash
   conda activate AutoQuant && python -m backtest.factor.admission reject <factor_id> \
     --notes "Abandoned after N rounds: <final_failure_type>"
   ```
   这会：
   - 从 `data/duckdb/factors_pending.duckdb` 删除该因子列
   - 在 `data/factor_library/registry.json` 中标记 `status="rejected"`
   - 记录 `admission_history`（含 notes 和 strategy_config）

2. **删除临时代码**（除非 `--keep-code`）：
   ```bash
   rm -rf alphas/exp/agent/<factor_id>/
   ```
   这移除：
   - `alphas/exp/agent/<factor_id>/factor.py`
   - `alphas/exp/agent/<factor_id>/config.yaml`

3. **归档 results 目录**：
   如果 `results/<factor_id>/` 存在，移入统一归档目录：
   ```bash
   mkdir -p results/rejected
   mv results/<factor_id>/ results/rejected/ 2>/dev/null || true
   ```
   同时清理对应的 `results/<factor_id>_run_xxx/` 追踪目录（如有）。

4. **KB 记录检查**（交互式确认）：
   检查以下文件是否已包含该因子的记录，如未记录则提示用户：
   - `agents/knowledge_base/failed_attempts.jsonl`
   - `agents/knowledge_base/anti_patterns.json`（如 RC 输出了 `new_anti_pattern`）

   检查命令：
   ```bash
   grep -q '<factor_id>' agents/knowledge_base/failed_attempts.jsonl && echo "✅ failed_attempts" || echo "❌ failed_attempts missing"
   grep -q '<factor_id>' agents/knowledge_base/anti_patterns.json && echo "✅ anti_patterns" || echo "❌ anti_patterns missing"
   ```

5. **保留归档**：
   `results/rejected/<factor_id>/` 保留完整实验档案：
   - `trace.jsonl` — 迭代记录
   - `*/pipeline_report.md` — 诊断报告
   - `*/plots/` — 图表
   - `*/result.json` — 结构化结果

6. **最终输出**：
   | 操作 | 状态 |
   |------|------|
   | work DB 清理 | ✅ / ❌ |
   | registry 标记 rejected | ✅ / ❌ |
   | 临时代码删除 | ✅ / ❌ (保留) |
   | results 移入 rejected/ | ✅ / ❌ |
   | KB 记录检查 | ✅ 已记录 / ⚠️ 未记录 |
   | rejected/ 归档保留 | ✅ |

## Implementation Notes

- `admission reject` **只能对未 admitted 的因子执行**。如果因子已被 admit，会报错，此时应使用 `unadmit` 而非 `reject`。
- `--keep-code` 用于需要保留代码供后续参考的场景（如提取部分逻辑到新因子）。
- 如果用户未提供 `factor_id`，列出所有 `status != "admitted"` 的因子供选择：
  ```bash
  conda activate AutoQuant && python -m backtest.factor.admission status
  ```

## Example

```text
/reject-factor f_auto_20260602_001
→ conda activate AutoQuant && python -m backtest.factor.admission reject f_auto_20260602_001 --notes "Abandoned after 6 rounds: backtest_fail. Alpha trapped in LS spread."
→ rm -rf alphas/exp/agent/f_auto_20260602_001/
→ mkdir -p results/rejected && mv results/f_auto_20260602_001/ results/rejected/
→ grep KB files for presence
→ print summary table
```
