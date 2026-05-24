"""Hypothesis generation and Hypothesis→Experiment conversion.

Uses DeepSeek API (OpenAI-compatible) to:
1. Generate novel factor hypotheses based on scenario context + history
2. Convert hypotheses into runnable Python code with @register decorators
"""

from __future__ import annotations

import ast
import json
import re
from typing import TYPE_CHECKING, Any

from .core.proposal import Hypothesis, Hypothesis2Experiment, HypothesisGen
from .core.utils import render_prompt
from .experiment import AutoQuantFactorExperiment

if TYPE_CHECKING:
    from .core.evolving_framework import Trace
    from .core.knowledge_base import KnowledgeBase
    from .scenario import AShareQuantScenario


try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _generate_factor_id(batch: str | None = None, seq: int | None = None) -> str:
    """Generate a unique factor ID for AI-generated factors.

    Format: ``f_auto_{batch}_{seq}`` — distinct from human factors ``f_###``.
    """
    import uuid

    if batch and seq is not None:
        return f"f_auto_{batch}_{seq:03d}"
    return f"f_auto_{uuid.uuid4().hex[:8]}"


def _extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from LLM response text.

    Handles markdown fences (```json ... ```) and raw JSON.
    """
    # Try fenced JSON first
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        # Try bare JSON object — balance braces to handle nesting
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found in text")
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    text = text[start : i + 1]
                    break
        else:
            raise ValueError("Unbalanced braces in JSON text")
    return json.loads(text)


def _extract_python_code(text: str) -> str:
    """Extract Python code from LLM response text.

    Handles markdown fences (```python ... ```) and raw code.
    Prefers explicitly tagged python blocks to avoid grabbing JSON fences.
    """
    # Prefer explicitly tagged python blocks
    fenced = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    # Fallback: any fenced block
    fenced = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _validate_python_code(code: str) -> None:
    """Basic syntax validation: parse with ast.

    Raises SyntaxError if the code is invalid Python.
    """
    ast.parse(code)


def _inject_factor_id(code: str, factor_id: str) -> str:
    """Replace the factor_id in the @register decorator using AST.

    The LLM may generate a placeholder ID; we force the actual generated ID.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Invalid code — prepend decorator and let caller validate later
        return f'@register("{factor_id}")\n' + code

    modified = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for i, dec in enumerate(node.decorator_list):
                # Handle @register("...") or @register(factor_id="...")
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "register":
                    if dec.args and isinstance(dec.args[0], ast.Constant):
                        dec.args[0] = ast.Constant(value=factor_id)
                        modified = True
                    elif dec.keywords:
                        for kw in dec.keywords:
                            if kw.arg == "factor_id" and isinstance(kw.value, ast.Constant):
                                kw.value = ast.Constant(value=factor_id)
                                modified = True
                                break
                    break
                # Handle bare @register (no parentheses)
                elif isinstance(dec, ast.Name) and dec.id == "register":
                    node.decorator_list[i] = ast.Call(
                        func=ast.Name(id="register", ctx=ast.Load()),
                        args=[ast.Constant(value=factor_id)],
                        keywords=[],
                    )
                    modified = True
                    break
            if modified:
                break

    if modified:
        tree = ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    # No @register found — prepend one
    return f'@register("{factor_id}")\n' + code


# ---------------------------------------------------------------------------
# Hypothesis Generator
# ---------------------------------------------------------------------------


class AutoQuantFactorHypothesisGen(HypothesisGen):
    """Generate factor hypotheses using DeepSeek LLM."""

    def __init__(
        self,
        scenario: "AShareQuantScenario",
        llm_client: Any,
        knowledge_base: "KnowledgeBase | None" = None,
    ):
        super().__init__(scenario)
        self.llm = llm_client
        self.kb = knowledge_base
        self._prompt_dir = scenario._prompt_dir

    def gen(
        self,
        trace: "Trace | None" = None,
        *,
        seed_hypothesis: Hypothesis | None = None,
    ) -> Hypothesis:
        """Generate a new factor hypothesis.

        Parameters
        ----------
        trace : Trace | None
            History of past experiments and feedback.
        seed_hypothesis : Hypothesis | None
            If provided, skip LLM generation and use this as the first-round
            hypothesis.  Useful when the user already has a concrete idea.

        Returns
        -------
        Hypothesis
        """
        if seed_hypothesis is not None:
            return seed_hypothesis

        if OpenAI is None:
            raise RuntimeError(
                "openai SDK is not installed. "
                "Install it with: pip install openai"
            )

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(trace)

        # Call DeepSeek via OpenAI-compatible API
        response = self.llm.chat.completions.create(
            model=getattr(self.llm, "_default_model", "deepseek-chat"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4096,
            temperature=0.7,
        )

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError("LLM response truncated (finish_reason='length'); increase max_tokens")
        content = choice.message.content
        if not content:
            raise RuntimeError("LLM returned empty content (possible refusal or empty completion)")
        data = _extract_json(content)

        return Hypothesis(
            hypothesis_text=data.get("hypothesis_text", ""),
            category=data.get("category", ""),
            data_sources=data.get("data_sources", []),
            rationale=data.get("rationale", ""),
            expected_behavior=data.get("expected_behavior", ""),
            keywords=data.get("keywords", []),
        )

    def _build_system_prompt(self) -> str:
        """Load and return the hypothesis generation system prompt."""
        path = self._prompt_dir / "hypothesis_gen.system.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        # Fallback minimal prompt
        return (
            "You are a quantitative researcher. Generate a novel A-share factor "
            "hypothesis in JSON format with keys: hypothesis_text, category, "
            "data_sources, rationale, expected_behavior, keywords."
        )

    def _build_user_prompt(self, trace: "Trace | None") -> str:
        """Render the user prompt with context, history, and KB."""
        # Scenario description
        scenario_desc = self.scenario.render_scenario_prompt()

        # History from trace
        history = self._format_history(trace)

        # KB retrieval
        kb_cases = ""
        if self.kb is not None:
            # We can't retrieve_similar without a hypothesis, so just show SOTA
            kb_cases = str(self.kb.get_sota())

        # SOTA
        sota = "No prior experiments."
        if trace and trace.hist:
            successes = trace.successes()
            if successes:
                best = max(successes, key=lambda x: x[1].metrics.get("rankicir", float("-inf")))
                sota = f"Best so far: {best[0].experiment_id} with RankICIR = {best[1].rankicir:.3f}"

        template_path = self._prompt_dir / "hypothesis_gen.user.md"
        if template_path.exists():
            return render_prompt(
                template_path,
                scenario_desc=scenario_desc,
                history=history,
                kb_cases=kb_cases,
                sota=sota,
            )

        # Fallback
        parts = ["## Scenario\n\n", scenario_desc, "\n\n## History\n\n", history]
        return "\n".join(parts)

    def _format_history(self, trace: "Trace | None") -> str:
        if not trace or not trace.hist:
            return "No previous experiments."

        lines: list[str] = []
        for i, (exp, fb) in enumerate(trace.hist[-5:], 1):  # Last 5 only
            status = "PASS" if fb.decision else "FAIL"
            parts = [f"{i}. {exp.experiment_id} [{status}]"]
            rankicir = fb.metrics.get("rankicir")
            ic_pos = fb.metrics.get("ic_positive_ratio")
            turnover = fb.metrics.get("turnover")
            if rankicir is not None:
                parts.append(f"RankICIR={rankicir:.3f}")
            if ic_pos is not None:
                parts.append(f"IC+={ic_pos:.1%}")
            if turnover is not None:
                parts.append(f"Turnover={turnover:.3f}")
            obs = getattr(fb, "observation", "")[:200]
            parts.append(f"\n   Observation: {obs}...")
            lines.append(" — ".join(parts))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hypothesis → Experiment Converter
# ---------------------------------------------------------------------------


class AutoQuantFactorHypothesis2Experiment(Hypothesis2Experiment):
    """Convert a Hypothesis into executable AutoQuant factor code."""

    def __init__(
        self,
        scenario: "AShareQuantScenario",
        llm_client: Any,
    ):
        super().__init__(scenario)
        self.llm = llm_client
        self._prompt_dir = scenario._prompt_dir

    def convert(
        self,
        hypothesis: Hypothesis,
        trace: "Trace | None" = None,
    ) -> AutoQuantFactorExperiment:
        """Convert a hypothesis into an executable experiment.

        Parameters
        ----------
        hypothesis : Hypothesis
        trace : Trace | None
            Ignored for now; reserved for future context-aware code generation.

        Returns
        -------
        AutoQuantFactorExperiment
        """
        if OpenAI is None:
            raise RuntimeError(
                "openai SDK is not installed. "
                "Install it with: pip install openai"
            )

        # Generate factor ID
        factor_id = _generate_factor_id()

        # Build code generation prompt
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(hypothesis)

        # Call DeepSeek via OpenAI-compatible API
        response = self.llm.chat.completions.create(
            model=getattr(self.llm, "_default_model", "deepseek-chat"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4096,
            temperature=0.3,  # Lower temperature for code generation
        )

        # Extract and validate code
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError("LLM code response truncated (finish_reason='length'); increase max_tokens")
        content = choice.message.content
        if not content:
            raise RuntimeError("LLM returned empty content (possible refusal or empty completion)")
        code = _extract_python_code(content)
        code = _inject_factor_id(code, factor_id)
        _validate_python_code(code)

        return AutoQuantFactorExperiment(
            factor_id=factor_id,
            factor_code=code,
        )

    def _build_system_prompt(self) -> str:
        path = self._prompt_dir / "hypothesis2experiment.system.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return (
            "You are a Python code generator for quantitative alpha factors. "
            "Generate a @register-decorated function that returns a pd.Series."
        )

    def _build_user_prompt(self, hypothesis: Hypothesis) -> str:
        template_path = self._prompt_dir / "hypothesis2experiment.user.md"

        # Build data-source-specific panel column documentation
        panel_cols = self.scenario.get_panel_columns_for_data_sources(
            hypothesis.data_sources
        )
        panel_lines: list[str] = []
        for src, cols in panel_cols.items():
            if src.startswith("_"):
                panel_lines.append(f"- **{src[1:]}**: {', '.join(cols)}")
            else:
                panel_lines.append(f"- **{src}**:")
                # Show first 10 + "..." to avoid overwhelming the prompt
                display = cols[:10]
                if len(cols) > 10:
                    display.append(f"... ({len(cols) - 10} more)")
                for c in display:
                    panel_lines.append(f"  - `{c}`")
        panel_columns_md = "\n".join(panel_lines) if panel_lines else (
            "_No data sources specified. Use only basic market daily columns "
            "(open, close, volume, etc.)._"
        )

        if template_path.exists():
            return render_prompt(
                template_path,
                hypothesis_text=hypothesis.hypothesis_text,
                category=hypothesis.category,
                data_sources=", ".join(hypothesis.data_sources),
                panel_columns=panel_columns_md,
                rationale=hypothesis.rationale,
                expected_behavior=hypothesis.expected_behavior,
            )

        # Fallback
        return (
            f"## Factor Hypothesis\n\n{hypothesis.hypothesis_text}\n\n"
            f"### Category\n{hypothesis.category}\n\n"
            f"### Data Sources\n{', '.join(hypothesis.data_sources)}\n\n"
            f"### Available Panel Columns\n{panel_columns_md}\n\n"
            f"Generate the Python implementation."
        )
