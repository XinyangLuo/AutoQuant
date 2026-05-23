## Factor Hypothesis

{{hypothesis_text}}

### Category
{{category}}

### Required Data Sources
{{data_sources}}

### Rationale
{{rationale}}

### Expected Behavior
{{expected_behavior}}

---

Generate the complete Python implementation. The function must:
1. Accept a `panel: pd.DataFrame` (and any parameters declared in `@register(..., parameters={...})`)
2. Return a `pd.Series` with the factor values
3. Set the DataFrame index to `(date, symbol)` before applying operators
4. Use only the operators and data columns listed above
5. Handle NaN values gracefully (most operators preserve NaN automatically)

Remember: The factor values will be post-processed by the neutralization pipeline (`barra_ind_size` by default), so you do NOT need to neutralize in the function itself.

Return ONLY the Python code, no extra text.
