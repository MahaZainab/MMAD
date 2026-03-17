# =============================================================================
# CommonsenseQA Multi-Model Evaluator  -  Peer Debate with Confidence Scores
#
# Dataset : CommonsenseQA  (each example has: id, question, question_concept,
#                           choices {label: [...], text: [...]}, answerKey)
#           The task is to select the single best answer (A-E) to a commonsense
#           multiple-choice question.
#
# Models:
#   M     = Mistral-7B-Instruct-v0.3         (student agent)
#   P     = Phi-4-mini-instruct               (student agent)
#   Q     = Qwen2.5-7B-Instruct               (student agent)
#   Judge = Qwen2.5-7B-Instruct               (evaluator, unchanged)
#
# Pipeline
# --------
# Phase 0      : Each agent independently answers the question AND provides
#                a confidence score (0.0 – 1.0).
#                --> Judge scores Phase 0
#
# Rounds 1-4   : Pure peer-debate (no teacher at any stage)
#
#     Step A  : Each agent sees its own previous answer + confidence score,
#               plus both peers' previous answer labels + confidence scores
#               ONLY (no reasoning shared — agents must think independently).
#               Agent then revises its answer and emits an updated confidence score.
#     Step B  : Judge scores revised answers.
#
# Output    : Single JSON + CSV with ALL rounds side-by-side.
#
# CSV columns
# -----------
# id, question, question_concept, choices, gold_answer,
# M_output,    M_confidence,    M_Correctness,    <- Phase 0
# P_output,    P_confidence,    P_Correctness,
# Q_output,    Q_confidence,    Q_Correctness,
# M_output_d1, M_confidence_d1, M_Correctness_d1, <- Round 1
# ...repeated for d2, d3, d4...
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
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_RESULTS_JSON   = "csqa_confidence_results.json"
DEFAULT_RESULTS_CSV    = "csqa_confidence_results.csv"

NUM_DEBATE_ROUNDS      = 4

# Student models
MODEL_M = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_P = "microsoft/Phi-4-mini-instruct"
MODEL_Q = "Qwen/Qwen2.5-7B-Instruct"

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
    return "\n".join(f"  {lbl}) {txt}" for lbl, txt in zip(labels, texts))


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def build_contestant_prompt(question: str, question_concept: str, choices_str: str) -> str:
    """Phase 0: independent answering with confidence score, no peer context."""
    return f"""You are answering a commonsense multiple-choice question.
The key concept in this question is: "{question_concept}".

Task:
Read the question and the answer choices carefully. Select the single best answer.
Return your chosen answer label and a confidence score reflecting how certain you are.

Definition of confidence_score:
- A float between 0.0 and 1.0 representing how confident you are in your answer.
- 1.0 = completely certain, 0.5 = unsure, 0.0 = guessing.

Rules for predicted_output (your answer):
- Return ONLY the single letter label (A, B, C, D, or E).

Return ONLY valid JSON with exactly these keys:
{{
  "predicted_output": string,
  "confidence_score": float
}}

Question:
{question}

Choices:
{choices_str}
"""


def build_debate_prompt(
    question: str,
    question_concept: str,
    choices_str: str,
    my_label: str,
    round_num: int,
    my_output: str,
    my_confidence: float,
    peer_responses: List[Tuple[str, str, float]],  # [(label, answer_label, confidence), ...]
) -> str:
    """
    Peer-debate prompt for CommonsenseQA.

    Peers share ONLY their answer label and confidence score — no reasoning.
    The agent must form its own independent judgment and then decide whether
    to revise based on what the peers chose and how confident they were.
    """
    peer_block = ""
    for peer_label, peer_output, peer_confidence in peer_responses:
        peer_block += (
            f"  Agent {peer_label} ({AGENT_DISPLAY_NAME[peer_label]}): "
            f"Answer={peer_output}  Confidence={peer_confidence:.2f}\n"
        )

    return f"""You are Agent {my_label} ({AGENT_DISPLAY_NAME[my_label]}), participating in \
Round {round_num} of a {NUM_DEBATE_ROUNDS}-round peer debate for commonsense QA.

You will receive:
1. The commonsense multiple-choice question and answer choices.
2. YOUR answer label and confidence score from the previous round.
3. The answer labels and confidence scores of the other two agents (NO reasoning — you must think independently).

Your job:
- Consider what each peer chose and how confident they were.
- A high-confidence peer answer deserves more weight, but do not blindly follow the majority.
- Re-examine the question using your own commonsense reasoning.
- If peer signals cause you to doubt your answer, revise it.
- If you remain confident, keep your answer.
- Update your confidence score to reflect your current certainty.

Rules for predicted_output (your answer):
- Return ONLY the single letter label (A, B, C, D, or E).

Return ONLY valid JSON with exactly these keys:
{{
  "predicted_output": string,
  "confidence_score": float
}}

=== Question (concept: "{question_concept}") ===
{question}

=== Choices ===
{choices_str}

=== Your Answer from Previous Round ===
  Agent {my_label}: Answer={my_output}  Confidence={my_confidence:.2f}

=== Peers' Answers from Previous Round (answer + confidence only) ===
{peer_block}
Now reconsider the question and return your (possibly revised) answer as valid JSON.
"""


def build_judge_prompt(question: str, choices_str: str, gold: str, prediction: str) -> str:
    return f"""You are an automated judge for a commonsense multiple-choice QA task.

Your job is to decide whether the predicted answer label is CORRECT compared to the gold answer label.

Rules:
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
      - JSONL (one JSON object per line) with keys:
        id, question, question_concept, choices {label: [...], text: [...]}, answerKey
      - JSON array of the same dicts
      - CSV with columns: id, question, question_concept, choices (JSON string), answerKey

    Normalises to internal keys:
      id, input (=question), question_concept, choices (dict), output (=answerKey)
    """
    rows = []

    if path.lower().endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                try:
                    choices = json.loads(row.get("choices", "{}"))
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
# Phase 0 - baseline: each agent independently answers with confidence
# ---------------------------------------------------------------------------
def run_contestant(
    label: str,
    model_name: str,
    dataset: List[Dict],
    max_new_tokens: int,
) -> List[Dict]:
    """
    Returns a list of per-example dicts:
      { "output": str, "confidence": float, "raw_text": str }
    """
    print(f"\n{'='*60}")
    print(f"PHASE 0 (Baseline) - Agent [{label}]  {model_name}")
    print(f"{'='*60}")

    runner  = ModelRunner(model_name)
    results = []

    for idx, ex in enumerate(dataset):
        inp         = ex.get("input",            "")
        concept     = ex.get("question_concept", "")
        choices_str = format_choices(ex.get("choices", {"label": [], "text": []}))
        ex_id       = ex.get("id",               f"idx_{idx}")

        prompt = build_contestant_prompt(inp, concept, choices_str)

        print(f"  [{label}] Example {idx+1}/{len(dataset)}  id={ex_id}", end="  ")
        raw = runner.generate(prompt, max_new_tokens)

        obj  = _first_json_object(raw)
        out  = str(obj.get("predicted_output", obj.get("output", "")) or "") if obj else ""
        conf = float(obj.get("confidence_score", 0.5)) if obj else 0.5
        conf = max(0.0, min(1.0, conf))   # clamp to [0, 1]

        # Normalise to single uppercase letter
        out = out.strip().upper()
        if out and out[0] in "ABCDE":
            out = out[0]
        else:
            out = ""

        results.append({"output": out, "confidence": conf, "raw_text": raw})
        print(f"-> predicted: {out!r}  conf={conf:.2f}")

    runner.unload()
    return results


# ---------------------------------------------------------------------------
# Debate round - each agent revises based on peers' answer + confidence only
# ---------------------------------------------------------------------------
def run_debate_round(
    label: str,
    model_name: str,
    dataset: List[Dict],
    prev_round_preds: Dict[str, List[Dict]],
    round_num: int,
    max_new_tokens: int,
) -> List[Dict]:
    """
    One pure peer-debate round for a single agent.

    Each agent sees its own previous answer + confidence, plus both peers'
    answer labels + confidence scores ONLY (no reasoning shared).

    Returns a list of per-example dicts:
      { "output": str, "confidence": float, "raw_text": str }
    """
    print(f"\n{'='*60}")
    print(f"DEBATE ROUND {round_num}/{NUM_DEBATE_ROUNDS} - Agent [{label}]  {model_name}")
    print(f"{'='*60}")

    runner  = ModelRunner(model_name)
    results = []

    for idx, ex in enumerate(dataset):
        inp         = ex.get("input",            "")
        concept     = ex.get("question_concept", "")
        choices_str = format_choices(ex.get("choices", {"label": [], "text": []}))
        ex_id       = ex.get("id",               f"idx_{idx}")

        my_output    = prev_round_preds[label][idx]["output"]
        my_confidence = prev_round_preds[label][idx]["confidence"]

        # Peers share answer label + confidence ONLY
        peer_responses: List[Tuple[str, str, float]] = []
        for peer_label, _ in PEER_LABELS[label]:
            peer_output    = prev_round_preds[peer_label][idx]["output"]
            peer_confidence = prev_round_preds[peer_label][idx]["confidence"]
            peer_responses.append((peer_label, peer_output, peer_confidence))

        prompt = build_debate_prompt(
            inp, concept, choices_str,
            my_label=label,
            round_num=round_num,
            my_output=my_output,
            my_confidence=my_confidence,
            peer_responses=peer_responses,
        )

        print(f"  [{label}] Example {idx+1}/{len(dataset)}  id={ex_id}", end="  ")
        raw = runner.generate(prompt, max_new_tokens)

        obj  = _first_json_object(raw)
        out  = str(obj.get("predicted_output", obj.get("output", "")) or "") if obj else ""
        conf = float(obj.get("confidence_score", 0.5)) if obj else 0.5
        conf = max(0.0, min(1.0, conf))

        # Normalise to single uppercase letter
        out = out.strip().upper()
        if out and out[0] in "ABCDE":
            out = out[0]
        else:
            out = ""

        # Fallback: retain previous answer if model returns nothing useful
        if not out:
            out  = my_output
            conf = my_confidence

        results.append({"output": out, "confidence": conf, "raw_text": raw})
        print(f"-> revised: {out!r}  conf={conf:.2f}")

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

    Returns { agent_label: [ "CORRECT" | "INCORRECT", ... ] }
    """
    print(f"\n{'='*60}")
    print(f"JUDGE  [{phase_label}]  model={MODEL_JUDGE}")
    print(f"{'='*60}")

    runner      = ModelRunner(MODEL_JUDGE)
    correctness: Dict[str, List[str]] = {lbl: [] for lbl in predictions}

    for idx, ex in enumerate(dataset):
        inp         = ex.get("input",   "")
        choices_str = format_choices(ex.get("choices", {"label": [], "text": []}))
        gold        = ex.get("output",  "").strip().upper()
        ex_id       = ex.get("id",      f"idx_{idx}")

        print(f"  [Judge] Example {idx+1}/{len(dataset)}  id={ex_id}")

        for lbl in predictions:
            pred = predictions[lbl][idx]["output"].strip().upper()

            # Fast exact-match shortcut
            if pred and gold and pred[0] == gold[0]:
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
                verdict = "INCORRECT"

            correctness[lbl].append(verdict)
            print(f"    [{lbl}] pred={pred!r}  gold={gold!r}  ->  {verdict}")

    runner.unload()
    return correctness


# ---------------------------------------------------------------------------
# Output writers - all phases in one JSON + CSV
# ---------------------------------------------------------------------------
def write_outputs(
    dataset: List[Dict],
    all_preds: List[Dict[str, List[Dict]]],   # index 0=Phase0, 1..4=rounds
    all_corr:  List[Dict[str, List[str]]],    # parallel correctness
    json_path: str = DEFAULT_RESULTS_JSON,
    csv_path:  str = DEFAULT_RESULTS_CSV,
) -> None:
    """
    Writes a single JSON + CSV with all phases side-by-side.

    Column naming:
      Phase 0 (baseline) : {lbl}_output,    {lbl}_confidence,    {lbl}_Correctness
      Debate round N     : {lbl}_output_dN, {lbl}_confidence_dN, {lbl}_Correctness_dN

    Note: no reasoning column — agents only share answer + confidence in this pipeline.
    """
    labels = ["M", "P", "Q"]
    rows   = []

    for idx, ex in enumerate(dataset):
        choices_str = format_choices(ex.get("choices", {"label": [], "text": []}))
        row = {
            "id":               ex.get("id",               ""),
            "question":         ex.get("input",            ""),
            "question_concept": ex.get("question_concept", ""),
            "choices":          choices_str,
            "gold_answer":      ex.get("output",           ""),
        }

        for phase_idx, (preds, corr) in enumerate(zip(all_preds, all_corr)):
            suffix = "" if phase_idx == 0 else f"_d{phase_idx}"
            for lbl in labels:
                p = preds.get(lbl, [])
                c = corr.get(lbl,  [])
                row[f"{lbl}_output{suffix}"]      = p[idx]["output"]     if idx < len(p) else ""
                row[f"{lbl}_confidence{suffix}"]  = p[idx]["confidence"] if idx < len(p) else ""
                row[f"{lbl}_Correctness{suffix}"] = c[idx]               if idx < len(c) else ""

        rows.append(row)

    # ---- JSON ---------------------------------------------------------------
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nJSON results saved -> {json_path}")

    # ---- CSV ----------------------------------------------------------------
    fieldnames = ["id", "question", "question_concept", "choices", "gold_answer"]
    for phase_idx in range(len(all_preds)):
        suffix = "" if phase_idx == 0 else f"_d{phase_idx}"
        for lbl in labels:
            fieldnames += [
                f"{lbl}_output{suffix}",
                f"{lbl}_confidence{suffix}",
                f"{lbl}_Correctness{suffix}",
            ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
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
        return "Phase-0 (base)" if r == 0 else f"Round-{r}"

    phase_names = [_round_label(r) for r in range(num_phases)]
    col_w       = 22

    sep = "=" * (8 + col_w * num_phases)
    print("\n" + sep)
    print("SUMMARY  (peer debate, answer+confidence only  -  CommonsenseQA)")
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
    print(f"Debate rounds  : {NUM_DEBATE_ROUNDS}  (pure peer-debate, answer+confidence only)")
    dataset = load_dataset(dataset_path)
    print(f"Loaded         : {len(dataset)} examples\n")

    all_preds: List[Dict[str, List[Dict]]] = []
    all_corr:  List[Dict[str, List[str]]]  = []

    # -----------------------------------------------------------------------
    # PHASE 0 - Baseline: each agent independently answers with confidence
    # -----------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# PHASE 0 - BASELINE (independent answers + confidence scores)")
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

    # -----------------------------------------------------------------------
    # ROUNDS 1 - NUM_DEBATE_ROUNDS  (pure peer-debate, answer+confidence only)
    # -----------------------------------------------------------------------
    prev_preds = phase0_preds

    for round_num in range(1, NUM_DEBATE_ROUNDS + 1):
        print("\n" + "#" * 60)
        print(f"# DEBATE ROUND {round_num} / {NUM_DEBATE_ROUNDS}  [PURE PEER DEBATE]")
        print("#" * 60)

        round_preds: Dict[str, List[Dict]] = {}
        for label, model_name in CONTESTANT_MODELS:
            round_preds[label] = run_debate_round(
                label, model_name, dataset,
                prev_round_preds=prev_preds,
                round_num=round_num,
                max_new_tokens=max_new_tokens,
            )

        round_corr = run_judge(
            dataset, round_preds,
            phase_label=f"Round {round_num} (pure debate)",
            max_new_tokens=128,
        )

        all_preds.append(round_preds)
        all_corr.append(round_corr)
        prev_preds = round_preds

    # -----------------------------------------------------------------------
    # Save all phases to JSON + CSV
    # -----------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("# SAVING RESULTS (all phases)")
    print("#" * 60)

    write_outputs(
        dataset,
        all_preds,
        all_corr,
        json_path=DEFAULT_RESULTS_JSON,
        csv_path=DEFAULT_RESULTS_CSV,
    )

    print_summary(all_corr)


if __name__ == "__main__":
    main()
