# =============================================================================
# CommonsenseQA Multi-Model Evaluator  -  Teacher-Guided Theory-of-Mind Debate
#
# Dataset : CommonsenseQA  (each example has: id, question, question_concept,
#                           choices {label, text}, answerKey)
#           The task is to answer a multiple-choice commonsense question by
#           selecting the single best answer from 5 options (A-E).
#
# Models:
#   M       = Mistral-7B-Instruct-v0.3         (student agent)
#   P       = Phi-4-mini-instruct               (student agent)
#   Q       = Qwen2.5-7B-Instruct               (student agent)
#   Teacher = Qwen3-Coder-30B-A3B-Instruct      (teacher / ToM guide, sees gold answer privately)
#   Judge   = Qwen2.5-7B-Instruct               (evaluator, unchanged)
#
# Pipeline
# --------
# Phase 0      : Each student independently answers the question (baseline)
#                --> Judge scores Phase 0
#
# Rounds 1-5   : Mixed schedule controlled by TEACHER_INTERVAL = 2
#
#   Pure-debate rounds  (round_num % TEACHER_INTERVAL != 0):
#     Step A  : Each student sees its own previous answer + both peers'
#               previous answers and revises freely (no teacher).
#     Step B  : Judge scores revised answers.
#
#   Teacher rounds  (round_num % TEACHER_INTERVAL == 0):
#     Step A  : Teacher reads ALL three students' previous answers and
#               produces targeted Theory-of-Mind guidance per agent
#               (highlights reasoning flaws WITHOUT revealing the answer).
#     Step B  : Each student receives peer answers + teacher's personal
#               guidance and revises its answer.
#     Step C  : Judge scores revised answers.
#
#   With TEACHER_INTERVAL=2 and NUM_DEBATE_ROUNDS=5 the schedule is:
#     Round 1 -> pure debate
#     Round 2 -> teacher intervenes
#     Round 3 -> pure debate
#     Round 4 -> teacher intervenes
#     Round 5 -> pure debate
#
# Output    : Single JSON + CSV with ALL rounds side-by-side.
#
# CSV columns
# -----------
# id, question, question_concept, choices, gold_answer,
# M_output,    M_reasoning,    M_Correctness,           <- Phase 0
# P_output,    P_reasoning,    P_Correctness,
# Q_output,    Q_reasoning,    Q_Correctness,
# M_output_d1, M_reasoning_d1, M_Correctness_d1,        <- Round 1
# P_output_d1, P_reasoning_d1, P_Correctness_d1,
# Q_output_d1, Q_reasoning_d1, Q_Correctness_d1,
# ...repeated for d2, d3, d4, d5...
# (teacher guidance stored in JSON only as nested dicts per round)
# =============================================================================

import csv
import json
import re
from typing import Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_DATASET        = "dataset.jsonl"          # CommonsenseQA JSONL file
DEFAULT_MAX_NEW_TOKENS = 384
DEFAULT_RESULTS_JSON   = "csqa_debate_results_slm.json"
DEFAULT_RESULTS_CSV    = "csqa_debate_results_slm.csv"

NUM_DEBATE_ROUNDS      = 5

# Teacher fires every TEACHER_INTERVAL rounds (e.g. 2 = rounds 2, 4, ...).
TEACHER_INTERVAL       = 2

# Student models
MODEL_M = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_P = "microsoft/Phi-4-mini-instruct"
MODEL_Q = "Qwen/Qwen2.5-7B-Instruct"

# Teacher model  (larger model that guides students via Theory-of-Mind)
MODEL_TEACHER = "Qwen/Qwen3-Coder-30B-A3B-Instruct"

# Judge model
MODEL_JUDGE = "Qwen/Qwen2.5-7B-Instruct"

CONTESTANT_MODELS = [
    ("M", MODEL_M),
    ("P", MODEL_P),
    ("Q", MODEL_Q),
]

# Display names used inside prompts
AGENT_DISPLAY_NAME = {
    "M": "Mistral-7B",
    "P": "Phi-4-mini",
    "Q": "Qwen2.5-7B",
}

# For each agent: its two peers  (label, display-name)
PEER_LABELS: Dict[str, List[Tuple[str, str]]] = {
    "M": [("P", "Phi-4-mini"),  ("Q", "Qwen2.5-7B")],
    "P": [("M", "Mistral-7B"),  ("Q", "Qwen2.5-7B")],
    "Q": [("M", "Mistral-7B"),  ("P", "Phi-4-mini")],
}


# ---------------------------------------------------------------------------
# Stopping criteria  -  halt once the first complete JSON object is generated
# ---------------------------------------------------------------------------
class StopAfterFirstJSONObject(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer, prompt_len: int):
        super().__init__()
        self.tokenizer  = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen_ids = input_ids[0, self.prompt_len:]
        if gen_ids.numel() == 0:
            return False
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        if "{" not in text:
            return False
        start = text.find("{")
        sub   = text[start:]
        depth, in_str, esc = 0, False, False
        for ch in sub:
            if esc:        esc = False;          continue
            if ch == "\\": esc = True;           continue
            if ch == '"':  in_str = not in_str;  continue
            if in_str:     continue
            if ch == "{":  depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return True
        return False


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------
def _first_json_object(text: str) -> Optional[dict]:
    """Extract the first valid JSON dict from raw model output."""
    if not text:
        return None
    t = text.strip()

    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    starts = [m.start() for m in re.finditer(r"\{", t)]
    for i in starts:
        for j in range(i + 1, len(t)):
            if t[j] != "}":
                continue
            try:
                obj = json.loads(t[i: j + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Helper: format choices into a readable block for prompts
# ---------------------------------------------------------------------------
def format_choices(choices: Dict) -> str:
    """
    choices = {"label": ["A","B","C","D","E"], "text": ["opt1","opt2",...]}
    Returns:
      A) opt1
      B) opt2
      ...
    """
    labels = choices.get("label", [])
    texts  = choices.get("text",  [])
    lines  = []
    for lbl, txt in zip(labels, texts):
        lines.append(f"  {lbl}) {txt}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def build_contestant_prompt(question: str, question_concept: str, choices_str: str) -> str:
    """Phase 0: independent answering for CommonsenseQA, no peer context."""
    return f"""You are answering a commonsense multiple-choice question.
The key concept in this question is: "{question_concept}".

Task:
Read the question and the answer choices carefully. Select the single best answer.
Return your chosen answer label (A, B, C, D, or E) along with a brief explanation.

Rules for predicted_output (your answer):
- Return ONLY the single letter label (A, B, C, D, or E).
- Do not return the full answer text — only the label.

Definition of explanation:
- A concise 2-3 sentence rationale explaining why you chose that answer.
- Reference commonsense reasoning or real-world knowledge where relevant.

Return ONLY valid JSON with exactly these keys:
{{
  "predicted_output": string,
  "explanation": string
}}

Question:
{question}

Choices:
{choices_str}
"""


def build_teacher_prompt(
    question: str,
    question_concept: str,
    choices_str: str,
    gold_answer: str,
    round_num: int,
    agent_responses: List[Tuple[str, str, str]],   # [(label, answer, reasoning), ...]
) -> str:
    """
    Teacher (Theory-of-Mind) prompt for CommonsenseQA.

    The teacher is privately given the correct gold answer label and uses it to
    produce targeted, Socratic guidance for EACH student individually.
    It must NOT reveal the gold answer label directly — instead it points out
    where each student's reasoning went wrong and steers them toward the correct answer.

    Returns JSON:
    {
      "guidance_M": "...",
      "guidance_P": "...",
      "guidance_Q": "..."
    }
    """
    agent_block = ""
    for label, answer, reasoning in agent_responses:
        agent_block += f"""
--- Agent {label} ({AGENT_DISPLAY_NAME[label]}) ---
Answer: {answer}
Reasoning: {reasoning}
"""

    return f"""You are an expert commonsense reasoning teacher and Theory-of-Mind reasoner \
participating in Round {round_num} of a {NUM_DEBATE_ROUNDS}-round multi-agent debate.

Three small language model agents (M, P, Q) are each trying to answer a commonsense \
multiple-choice question. The key concept is: "{question_concept}".
You have received their latest answers and reasoning.

[PRIVATE — FOR YOUR EYES ONLY]
The verified correct answer label is: "{gold_answer}"
Use this knowledge to assess each agent's answer with certainty.
Do NOT directly reveal this label in your guidance. Instead, use it to:
  - Confirm agents who are correct.
  - Precisely identify the reasoning flaw in agents who are wrong.
  - Steer incorrect agents firmly toward the right direction using commonsense cues.

Your role - Theory-of-Mind guidance:
- For EACH agent, compare their chosen label against the correct answer above.
- Identify the specific assumption or reasoning step that led them astray (or confirm correctness).
- If an agent is WRONG: Explain which aspect of commonsense or real-world knowledge they missed.
  Guide them firmly toward the correct answer without quoting the label directly.
- If an agent is CORRECT: Say "Your answer is on the right track.
  Maintain your position confidently even if peers disagree."

Return ONLY valid JSON with exactly these keys (one guidance string per agent):
{{
  "guidance_M": string,
  "guidance_P": string,
  "guidance_Q": string
}}

=== Question ===
{question}

=== Choices ===
{choices_str}

=== Agent Answers (Round {round_num - 1} output) ===
{agent_block}
"""


def build_debate_prompt(
    question: str,
    question_concept: str,
    choices_str: str,
    my_label: str,
    round_num: int,
    my_output: str,
    my_reasoning: str,
    peer_responses: List[Tuple[str, str, str]],   # [(label, answer, reasoning), ...]
    teacher_guidance: str,                          # "" on pure-debate rounds
) -> str:
    """
    Student debate prompt for CommonsenseQA - works for both pure-debate and teacher-guided rounds.
    """
    peer_block = ""
    for peer_label, peer_output, peer_reasoning in peer_responses:
        peer_block += f"""
--- Agent {peer_label} ({AGENT_DISPLAY_NAME[peer_label]}) ---
Answer: {peer_output}
Reasoning: {peer_reasoning}
"""

    if teacher_guidance.strip():
        teacher_section = f"""
=== Teacher Guidance (personalised for Agent {my_label}) ===
{teacher_guidance}
"""
        teacher_job_line = (
            "- Read the teacher's guidance carefully - it is targeted specifically at "
            "your reasoning.\n"
            "- Re-examine the question and choices, paying attention to the specific "
            "commonsense concern the teacher raised.\n"
        )
        closing = (
            "Now carefully reconsider the question using the teacher's guidance and return "
            "your (possibly revised) answer as valid JSON."
        )
    else:
        teacher_section  = ""
        teacher_job_line = ""
        closing = (
            "Now carefully reconsider the question and return your "
            "(possibly revised) answer as valid JSON."
        )

    return f"""You are Agent {my_label} ({AGENT_DISPLAY_NAME[my_label]}), participating in \
Round {round_num} of a {NUM_DEBATE_ROUNDS}-round multi-agent debate for commonsense QA.

You will receive:
1. The commonsense multiple-choice question and answer choices.
2. YOUR answer and reasoning from the previous round (which may be correct or incorrect).
3. The answers and reasoning of the other two agents from the previous round.

Your job:
{teacher_job_line}- Consider the other agents' reasoning critically - they may be right or wrong.
- Do NOT blindly follow the majority. A lone agent can be right while two are wrong.
- If another agent's reasoning exposes a flaw in your answer, revise it.
- If you remain confident in your answer, keep it and clearly explain why.

Rules for predicted_output (your answer):
- Return ONLY the single letter label (A, B, C, D, or E).
- Do not return the full answer text — only the label.

Return ONLY valid JSON with exactly these keys:
{{
  "predicted_output": string,
  "explanation": string
}}

=== Question (concept: "{question_concept}") ===
{question}

=== Choices ===
{choices_str}

=== Your Answer from Previous Round (Agent {my_label}) ===
Answer: {my_output}
Reasoning: {my_reasoning}

=== Other Agents' Answers from Previous Round ===
{peer_block}{teacher_section}
{closing}
"""


def build_judge_prompt(question: str, choices_str: str, gold: str, prediction: str) -> str:
    return f"""You are an automated judge for a commonsense multiple-choice QA task.

Your job is to decide whether the predicted answer label is CORRECT compared to the gold answer label.

Rules:
- Both gold and predicted answers should be single letter labels (A, B, C, D, or E).
- CORRECT if the predicted label matches the gold label exactly (case-insensitive).
- INCORRECT otherwise.
- If the predicted answer contains text instead of a label, try to infer the label from
  the choices and compare. If you cannot determine the label, mark INCORRECT.
- Return ONLY valid JSON with exactly these keys:
{{
  "verdict": "CORRECT" or "INCORRECT"
}}

Question:
{question}

Choices:
{choices_str}

Gold Answer Label:
{gold}

Predicted Answer Label:
{prediction}
"""


# ---------------------------------------------------------------------------
# Model wrapper  (loads once, generates many times, then unloads)
# ---------------------------------------------------------------------------
class ModelRunner:
    def __init__(self, model_name: str):
        print(f"  Loading: {model_name}")
        self.model_name = model_name
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)
        self.model      = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        print(f"  Loaded:  {model_name}")

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        inputs     = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[-1]
        stopping   = StoppingCriteriaList(
            [StopAfterFirstJSONObject(self.tokenizer, prompt_len)]
        )
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping,
            )
        gen_ids = outputs[0][prompt_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def unload(self):
        """Free GPU memory after use."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------
def load_dataset(path: str) -> List[Dict]:
    """
    Reads a CommonsenseQA dataset.
    Supports:
      - JSONL (one JSON object per line) with keys: id, question, question_concept,
        choices {label: [...], text: [...]}, answerKey
      - JSON (list of the same dicts)
      - CSV with columns: id, question, question_concept, choices (JSON string), answerKey

    Normalises to internal keys:
      id, input (=question), question_concept, choices (dict), output (=answerKey)
    """
    rows = []

    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                choices_raw = row.get("choices", "{}")
                try:
                    choices = json.loads(choices_raw)
                except Exception:
                    choices = {"label": [], "text": []}
                rows.append({
                    "id":               row.get("id", f"idx_{i}"),
                    "input":            row.get("question", row.get("input", "")),
                    "question_concept": row.get("question_concept", ""),
                    "choices":          choices,
                    "output":           row.get("answerKey", row.get("output", "")),
                })
        return rows

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Try JSONL first (one object per line)
    if content.startswith("{"):
        raw_list = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw_list.append(json.loads(line))
            except Exception:
                continue
    else:
        # Fallback: try loading as a JSON array
        raw_list = json.loads(content)
        if not isinstance(raw_list, list):
            raise ValueError("Dataset must be a JSONL file or a JSON array.")

    for i, ex in enumerate(raw_list):
        rows.append({
            "id":               ex.get("id", f"idx_{i}"),
            "input":            ex.get("question", ex.get("input", "")),
            "question_concept": ex.get("question_concept", ""),
            "choices":          ex.get("choices", {"label": [], "text": []}),
            "output":           ex.get("answerKey", ex.get("output", "")),
        })

    return rows


# ---------------------------------------------------------------------------
# Phase 0 - baseline: each agent independently answers
# ---------------------------------------------------------------------------
def run_contestant(
    label: str,
    model_name: str,
    dataset: List[Dict],
    max_new_tokens: int,
) -> List[Dict]:
    """
    Returns a list of per-example dicts:
      { "output": str, "reasoning": str, "raw_text": str }
    """
    print(f"\n{'='*60}")
    print(f"PHASE 0 (Baseline) - Agent [{label}]  {model_name}")
    print(f"{'='*60}")

    runner  = ModelRunner(model_name)
    results = []

    for idx, ex in enumerate(dataset):
        inp         = ex.get("input",            "")
        concept     = ex.get("question_concept", "")
        choices     = ex.get("choices",          {"label": [], "text": []})
        choices_str = format_choices(choices)
        ex_id       = ex.get("id",               f"idx_{idx}")

        prompt = build_contestant_prompt(inp, concept, choices_str)

        print(f"  [{label}] Example {idx+1}/{len(dataset)}  id={ex_id}", end="  ")
        raw = runner.generate(prompt, max_new_tokens)

        obj  = _first_json_object(raw)
        out  = str(obj.get("predicted_output", obj.get("output", "")) or "") if obj else ""
        expl = str(obj.get("explanation", "") or "") if obj else ""

        # Normalise output to uppercase letter only
        out = out.strip().upper()
        if out and out[0] in "ABCDE":
            out = out[0]

        results.append({"output": out, "reasoning": expl, "raw_text": raw})
        print(f"-> predicted: {out!r}")

    runner.unload()
    return results


# ---------------------------------------------------------------------------
# Teacher pass - generates per-agent Theory-of-Mind guidance for one round
# ---------------------------------------------------------------------------
def run_teacher(
    dataset: List[Dict],
    prev_round_preds: Dict[str, List[Dict]],
    round_num: int,
    max_new_tokens: int = 512,
) -> List[Dict[str, str]]:
    """
    For each example, the teacher is privately given the gold answer label and
    each agent's latest answer, then produces personalised guidance per agent.

    Returns a list (one entry per example) of dicts:
      {
        "guidance_M": str,
        "guidance_P": str,
        "guidance_Q": str,
        "raw_text":   str,
      }
    """
    print(f"\n{'='*60}")
    print(f"TEACHER PASS  Round {round_num}  model={MODEL_TEACHER}")
    print(f"{'='*60}")

    runner  = ModelRunner(MODEL_TEACHER)
    results = []

    for idx, ex in enumerate(dataset):
        inp         = ex.get("input",            "")
        concept     = ex.get("question_concept", "")
        choices     = ex.get("choices",          {"label": [], "text": []})
        choices_str = format_choices(choices)
        gold_answer = ex.get("output",           "")   # private gold answer label for teacher only
        ex_id       = ex.get("id",               f"idx_{idx}")

        # Collect all three agents' latest answers
        agent_responses: List[Tuple[str, str, str]] = []
        for lbl, _ in CONTESTANT_MODELS:
            out  = prev_round_preds[lbl][idx]["output"]
            expl = prev_round_preds[lbl][idx]["reasoning"]
            agent_responses.append((lbl, out, expl))

        prompt = build_teacher_prompt(inp, concept, choices_str, gold_answer, round_num, agent_responses)

        print(f"  [Teacher] Example {idx+1}/{len(dataset)}  id={ex_id}", end="  ")
        raw = runner.generate(prompt, max_new_tokens)

        obj = _first_json_object(raw)

        fallback = (
            "Re-examine the question carefully. Think about what makes practical, "
            "real-world sense given the key concept. Make sure you are not overthinking it."
        )
        g_M = str(obj.get("guidance_M", fallback) or fallback) if obj else fallback
        g_P = str(obj.get("guidance_P", fallback) or fallback) if obj else fallback
        g_Q = str(obj.get("guidance_Q", fallback) or fallback) if obj else fallback

        results.append({
            "guidance_M": g_M,
            "guidance_P": g_P,
            "guidance_Q": g_Q,
            "raw_text":   raw,
        })
        print(f"-> guidance generated (M:{len(g_M)}c  P:{len(g_P)}c  Q:{len(g_Q)}c)")

    runner.unload()
    return results


# ---------------------------------------------------------------------------
# Debate round - each agent revises its answer guided by teacher + peers
# ---------------------------------------------------------------------------
def run_debate_round(
    label: str,
    model_name: str,
    dataset: List[Dict],
    prev_round_preds: Dict[str, List[Dict]],
    teacher_guidance: Optional[List[Dict[str, str]]],  # None on pure-debate rounds
    round_num: int,
    max_new_tokens: int,
) -> List[Dict]:
    """
    One debate round for a single student agent.

    If teacher_guidance is not None (teacher round), each agent's prompt
    includes the teacher's personalised Theory-of-Mind hint.
    If teacher_guidance is None (pure-debate round), agents debate freely
    using only their own + peers' previous answers.

    Returns a list of per-example dicts:
      { "output": str, "reasoning": str, "raw_text": str }
    """
    print(f"\n{'='*60}")
    print(f"DEBATE ROUND {round_num}/{NUM_DEBATE_ROUNDS} - Agent [{label}]  {model_name}")
    print(f"{'='*60}")

    runner  = ModelRunner(model_name)
    results = []

    for idx, ex in enumerate(dataset):
        inp         = ex.get("input",            "")
        concept     = ex.get("question_concept", "")
        choices     = ex.get("choices",          {"label": [], "text": []})
        choices_str = format_choices(choices)
        ex_id       = ex.get("id",               f"idx_{idx}")

        # This agent's own answer from the previous round
        my_output    = prev_round_preds[label][idx]["output"]
        my_reasoning = prev_round_preds[label][idx]["reasoning"]

        # Both peers' answers from the previous round
        peer_responses: List[Tuple[str, str, str]] = []
        for peer_label, peer_name in PEER_LABELS[label]:
            peer_output    = prev_round_preds[peer_label][idx]["output"]
            peer_reasoning = prev_round_preds[peer_label][idx]["reasoning"]
            peer_responses.append((peer_label, peer_output, peer_reasoning))

        # Teacher's personalised guidance for this agent (empty on pure-debate rounds)
        t_guidance = (
            teacher_guidance[idx].get(f"guidance_{label}", "")
            if teacher_guidance is not None else ""
        )

        prompt = build_debate_prompt(
            inp, concept, choices_str,
            my_label=label,
            round_num=round_num,
            my_output=my_output,
            my_reasoning=my_reasoning,
            peer_responses=peer_responses,
            teacher_guidance=t_guidance,
        )

        print(f"  [{label}] Example {idx+1}/{len(dataset)}  id={ex_id}", end="  ")
        raw = runner.generate(prompt, max_new_tokens)

        obj  = _first_json_object(raw)
        out  = str(obj.get("predicted_output", obj.get("output", "")) or "") if obj else ""
        expl = str(obj.get("explanation", "") or "") if obj else ""

        # Normalise output to uppercase letter only
        out = out.strip().upper()
        if out and out[0] in "ABCDE":
            out = out[0]

        # Fallback: retain previous answer if model returns nothing useful
        if not out.strip():
            out  = my_output
            expl = f"[round-{round_num} fallback] {my_reasoning}"

        results.append({"output": out, "reasoning": expl, "raw_text": raw})
        print(f"-> revised: {out!r}")

    runner.unload()
    return results


# ---------------------------------------------------------------------------
# Judge - scores one full set of agent predictions
# ---------------------------------------------------------------------------
def run_judge(
    dataset: List[Dict],
    predictions: Dict[str, List[Dict]],
    phase_label: str = "",
    max_new_tokens: int = 128,
) -> Dict[str, List[str]]:
    """
    Evaluates every agent's predictions for a given phase/round.

    predictions  - { agent_label: [ {output, reasoning, raw_text}, ... ] }
    phase_label  - human-readable tag for logs (e.g. "Phase 0", "Round 3")

    Returns { agent_label: [ "CORRECT" | "INCORRECT", ... ] }
    """
    print(f"\n{'='*60}")
    print(f"JUDGE  [{phase_label}]  model={MODEL_JUDGE}")
    print(f"{'='*60}")

    runner      = ModelRunner(MODEL_JUDGE)
    correctness: Dict[str, List[str]] = {lbl: [] for lbl in predictions}

    for idx, ex in enumerate(dataset):
        inp         = ex.get("input",   "")
        choices     = ex.get("choices", {"label": [], "text": []})
        choices_str = format_choices(choices)
        gold        = ex.get("output",  "").strip().upper()
        ex_id       = ex.get("id",      f"idx_{idx}")

        print(f"  [Judge] Example {idx+1}/{len(dataset)}  id={ex_id}")

        for lbl in predictions:
            pred = predictions[lbl][idx]["output"].strip().upper()

            # Fast exact-match shortcut (avoids LLM call for clear cases)
            if pred and gold:
                if pred[0] == gold[0]:
                    verdict = "CORRECT"
                    print(f"    [{lbl}] pred={pred!r}  ->  {verdict}  (exact match)")
                    correctness[lbl].append(verdict)
                    continue

            # Empty-output guard
            if not pred:
                verdict = "CORRECT" if not gold else "INCORRECT"
                tag     = "(empty pred, gold also empty)" if not gold else "(empty pred)"
                print(f"    [{lbl}] pred=''  ->  {verdict}  {tag}")
                correctness[lbl].append(verdict)
                continue

            prompt  = build_judge_prompt(inp, choices_str, gold, pred)
            raw     = runner.generate(prompt, max_new_tokens)

            obj     = _first_json_object(raw)
            verdict = (obj.get("verdict", "") or "").upper() if obj else ""
            if verdict not in ("CORRECT", "INCORRECT"):
                verdict = "INCORRECT"   # safe default on parse failure

            correctness[lbl].append(verdict)
            print(f"    [{lbl}] pred={pred!r}  gold={gold!r}  ->  {verdict}")

    runner.unload()
    return correctness


# ---------------------------------------------------------------------------
# Output writers - all phases in one JSON + CSV
# ---------------------------------------------------------------------------
def write_outputs(
    dataset: List[Dict],
    all_preds:    List[Dict[str, List[Dict]]],            # index 0=Phase0, 1..5=rounds
    all_corr:     List[Dict[str, List[str]]],             # parallel correctness
    all_guidance: List[Optional[List[Dict[str, str]]]],   # teacher guidance per round (None for Phase 0)
    json_path: str = DEFAULT_RESULTS_JSON,
    csv_path:  str = DEFAULT_RESULTS_CSV,
) -> None:
    """
    Writes a single JSON + CSV with all phases side-by-side.

    Column naming:
      Phase 0 (baseline) : {lbl}_output,    {lbl}_reasoning,    {lbl}_Correctness
      Debate round N     : {lbl}_output_dN, {lbl}_reasoning_dN, {lbl}_Correctness_dN

    Teacher guidance is stored in JSON only (not CSV) to keep the CSV manageable.
    """
    labels = ["M", "P", "Q"]
    rows   = []

    for idx, ex in enumerate(dataset):
        choices     = ex.get("choices", {"label": [], "text": []})
        choices_str = format_choices(choices)
        row = {
            "id":               ex.get("id",               ""),
            "question":         ex.get("input",            ""),
            "question_concept": ex.get("question_concept", ""),
            "choices":          choices_str,
            "gold_answer":      ex.get("output",           ""),
        }

        # Student predictions + correctness for every phase
        for phase_idx, (preds, corr) in enumerate(zip(all_preds, all_corr)):
            suffix = "" if phase_idx == 0 else f"_d{phase_idx}"
            for lbl in labels:
                p = preds.get(lbl, [])
                c = corr.get(lbl,  [])
                row[f"{lbl}_output{suffix}"]      = p[idx]["output"]    if idx < len(p) else ""
                row[f"{lbl}_reasoning{suffix}"]   = p[idx]["reasoning"] if idx < len(p) else ""
                row[f"{lbl}_Correctness{suffix}"] = c[idx]              if idx < len(c) else ""

        # Teacher guidance (JSON only, stored as nested dict per round)
        for round_idx, guidance_list in enumerate(all_guidance):
            if guidance_list is None:
                continue   # Phase 0 has no teacher guidance
            rn = round_idx
            row[f"teacher_guidance_d{rn}"] = {
                "guidance_M": guidance_list[idx].get("guidance_M", ""),
                "guidance_P": guidance_list[idx].get("guidance_P", ""),
                "guidance_Q": guidance_list[idx].get("guidance_Q", ""),
            }

        rows.append(row)

    # ---- JSON ---------------------------------------------------------------
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nJSON results saved -> {json_path}")

    # ---- CSV  (student columns only; teacher guidance too nested for flat CSV) ----
    fieldnames = ["id", "question", "question_concept", "choices", "gold_answer"]
    for phase_idx in range(len(all_preds)):
        suffix = "" if phase_idx == 0 else f"_d{phase_idx}"
        for lbl in labels:
            fieldnames += [
                f"{lbl}_output{suffix}",
                f"{lbl}_reasoning{suffix}",
                f"{lbl}_Correctness{suffix}",
            ]

    # Flatten teacher guidance into CSV-friendly columns
    for round_num in range(1, NUM_DEBATE_ROUNDS + 1):
        for lbl in labels:
            fieldnames.append(f"teacher_guidance_{lbl}_d{round_num}")

    # Rebuild rows with flat teacher guidance for CSV
    flat_rows = []
    for idx, row in enumerate(rows):
        flat_row = {k: v for k, v in row.items() if not isinstance(v, dict)}
        for round_num in range(1, NUM_DEBATE_ROUNDS + 1):
            nested = row.get(f"teacher_guidance_d{round_num}", {})
            for lbl in labels:
                flat_row[f"teacher_guidance_{lbl}_d{round_num}"] = nested.get(f"guidance_{lbl}", "")
        flat_rows.append(flat_row)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"CSV results saved  -> {csv_path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(
    all_corr: List[Dict[str, List[str]]],
    labels: List[str] = ("M", "P", "Q"),
) -> None:
    num_phases  = len(all_corr)

    def _round_label(r):
        if r == 0:
            return "Phase-0 (base)"
        return f"Round-{r}[T]" if r % TEACHER_INTERVAL == 0 else f"Round-{r}[D]"

    phase_names = [_round_label(r) for r in range(num_phases)]
    col_w       = 22

    sep = "=" * (8 + col_w * num_phases)
    print("\n" + sep)
    print("SUMMARY  (teacher-guided ToM debate  -  CommonsenseQA)")
    print(sep)
    header = f"{'Agent':<8}" + "".join(f"{n:>{col_w}}" for n in phase_names)
    print(header)
    print("-" * (8 + col_w * num_phases))

    for lbl in labels:
        row_str  = f"  [{lbl}] "
        prev_pct = None
        for corr in all_corr:
            verdicts = corr.get(lbl, [])
            total    = len(verdicts)
            correct  = sum(1 for v in verdicts if v == "CORRECT")
            pct      = correct / total * 100 if total else 0.0

            if prev_pct is None:
                delta_str = ""
            else:
                diff  = pct - prev_pct
                arrow = "UP" if diff > 0 else ("DN" if diff < 0 else "--")
                delta_str = f"[{arrow}{abs(diff):.1f}%]"

            cell    = f"{correct}/{total}({pct:.1f}%) {delta_str}"
            row_str += f"{cell:>{col_w}}"
            prev_pct = pct

        print(row_str)

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    dataset_path   = DEFAULT_DATASET
    max_new_tokens = DEFAULT_MAX_NEW_TOKENS

    print(f"Dataset        : {dataset_path}  [CommonsenseQA]")
    print(f"Debate rounds  : {NUM_DEBATE_ROUNDS}")
    print(f"Teacher model  : {MODEL_TEACHER}")
    dataset = load_dataset(dataset_path)
    print(f"Loaded         : {len(dataset)} examples\n")

    # Accumulators for all phases
    all_preds:    List[Dict[str, List[Dict]]]           = []
    all_corr:     List[Dict[str, List[str]]]            = []
    all_guidance: List[Optional[List[Dict[str, str]]]]  = []

    # -----------------------------------------------------------------------
    # PHASE 0 - Baseline: each agent predicts independently (no teacher)
    # -----------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# PHASE 0 - BASELINE (independent answers)")
    print("#" * 60)

    phase0_preds: Dict[str, List[Dict]] = {}
    for label, model_name in CONTESTANT_MODELS:
        phase0_preds[label] = run_contestant(label, model_name, dataset, max_new_tokens)

    phase0_corr = run_judge(
        dataset, phase0_preds,
        phase_label="Phase 0 (baseline)",
        max_new_tokens=128,
    )

    all_preds.append(phase0_preds)
    all_corr.append(phase0_corr)
    all_guidance.append(None)       # no teacher in Phase 0

    # -----------------------------------------------------------------------
    # ROUNDS 1 - NUM_DEBATE_ROUNDS  (mixed schedule)
    # -----------------------------------------------------------------------
    prev_preds = phase0_preds

    for round_num in range(1, NUM_DEBATE_ROUNDS + 1):
        is_teacher_round = (round_num % TEACHER_INTERVAL == 0)
        round_type       = "TEACHER-GUIDED" if is_teacher_round else "PURE DEBATE"

        print("\n" + "#" * 60)
        print(f"# DEBATE ROUND {round_num} / {NUM_DEBATE_ROUNDS}  [{round_type}]")
        print("#" * 60)

        if is_teacher_round:
            teacher_guidance = run_teacher(
                dataset,
                prev_round_preds=prev_preds,
                round_num=round_num,
                max_new_tokens=512,
            )
        else:
            teacher_guidance = None

        round_preds: Dict[str, List[Dict]] = {}
        for label, model_name in CONTESTANT_MODELS:
            round_preds[label] = run_debate_round(
                label, model_name, dataset,
                prev_round_preds=prev_preds,
                teacher_guidance=teacher_guidance,
                round_num=round_num,
                max_new_tokens=max_new_tokens,
            )

        round_corr = run_judge(
            dataset, round_preds,
            phase_label=f"Round {round_num} ({round_type})",
            max_new_tokens=128,
        )

        all_preds.append(round_preds)
        all_corr.append(round_corr)
        all_guidance.append(teacher_guidance)

        prev_preds = round_preds

    # -----------------------------------------------------------------------
    # Save all phases + teacher guidance to JSON + CSV
    # -----------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# SAVING RESULTS (all phases + teacher guidance)")
    print("#" * 60)

    write_outputs(
        dataset,
        all_preds,
        all_corr,
        all_guidance,
        json_path=DEFAULT_RESULTS_JSON,
        csv_path=DEFAULT_RESULTS_CSV,
    )

    print_summary(all_corr)


if __name__ == "__main__":
    main()
