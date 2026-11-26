"""OneJev letter readout (policy.readout = letters; align with OmniJev/OneJev-4B, design B).

A DecisionInput is rendered in OneJev's qev format (github.com/OmniJev/OneJev, qev/prompt.py, PROMPT_VERSION
qev-labels-v2): the system prompt, then the user turn "<state>\\n{state}\\n</state>\\n\\n{question suffix}", then the
generation prompt with thinking off. The state = every prefix segment except the last (the decision question); images
stay where they are (vision blocks inside the state). One decision = one question; a grouped decision (payload "group")
= one question per group, all branching off the same state prefix. Each question lists its LEGAL candidates as options
"A. <id>: <text>" (two-letter labels after Z, as qev); the choice is read from the output-embedding rows of the option
labels at the last position (Qwen3.5 / OneJev tie them to the input embeddings).
"""

SYSTEM_PROMPT = ("Apply the question to the state. Choose exactly one of the listed options. "
                 "Respond with only its uppercase letter, with no explanation or reasoning.")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_NOT_ONE_TOKEN = {"BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV", "FZ", "GJ", "GK", "GQ", "GZ",
                  "HJ", "IY", "JF", "JG", "JH", "JL", "JN", "JQ"}
SLOT_LABELS = tuple(list(LETTERS) + [a + b for a in LETTERS for b in LETTERS if a + b not in _NOT_ONE_TOKEN][:256 - 26])
QUESTION_MARK = "Question type: choice\nQuestion:\n"
_STATE_MARK, _SUFFIX_MARK = "QEVSTATEMARK", "QEVSUFFIXMARK"


def chat_parts(tokenizer):
    """(head, tail): the chat-template text before the state body ("...<|im_start|>user\\n<state>\\n") and after the
    question suffix ("<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n")."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<state>\n{_STATE_MARK}\n</state>\n\n{_SUFFIX_MARK}"}]
    s = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    head, rest = s.split(_STATE_MARK)
    mid, tail = rest.split(_SUFFIX_MARK)
    assert mid == "\n</state>\n\n", repr(mid)
    return head, tail


def slot_token_ids(tokenizer, n=None):
    """Token id of every slot label that is one round-trip token (A..Z required; two-letter labels all or none)."""
    def one(label):
        enc = tokenizer.encode(label, add_special_tokens=False)
        return enc[0] if len(enc) == 1 and tokenizer.decode(enc) == label else None
    ids = [one(x) for x in LETTERS]
    assert all(i is not None for i in ids), "A..Z must be single tokens"
    extra = [one(x) for x in SLOT_LABELS[26:]]
    if all(i is not None for i in extra):
        ids += extra
    return ids if n is None else ids[:n]


def option_name_text(candidate_segments, cid):
    """Our candidate "Candidate:\\n<id>: <text>\\nDecision:" (environment.nanojev_candidate) -> (name, description)."""
    s = "".join(x for x in candidate_segments if isinstance(x, str)).strip()
    if s.startswith("Candidate:"):
        s = s[len("Candidate:"):].strip()
    if s.endswith("Decision:"):
        s = s[:-len("Decision:")].strip()
    if s.startswith(f"{cid}:"):
        s = s[len(cid) + 1:].strip()
    return cid, s


def question_suffix(instruction, options):
    """options: [(name, text)] in slot order -> the qev question suffix (tagged layout, names shown)."""
    labels = SLOT_LABELS[:len(options)]
    lines = "\n".join(f"{lab}. {name}: {text}" if text else f"{lab}. {name}" for lab, (name, text) in zip(labels, options))
    answer = ("Answer with one letter: {}." if len(options) <= 26 else "Answer with one label: {}.").format(", ".join(labels))
    return f"Question: {instruction}\n\nOptions:\n{lines}\n\n{answer}"


def questions_of(d):
    """[(instruction, [candidate index, ...])] of a DecisionInput: one question, or one per group (payload "group",
    instruction from d.group_questions[g] when given). Only legal candidates become options."""
    last = d.prefix[-1]
    assert isinstance(last, str) and last.startswith(QUESTION_MARK), "the last prefix segment must be the question"
    default = last[len(QUESTION_MARK):].strip()
    pls = d.payloads or []
    legal = set(d.legal_indices())
    if pls and all(isinstance(p, dict) and "group" in p for p in pls):
        order = list(dict.fromkeys(p["group"] for p in pls))
        gq = getattr(d, "group_questions", None) or {}
        out = []
        for g in order:
            idx = [i for i, p in enumerate(pls) if p["group"] == g and i in legal]
            if idx:
                text = gq.get(str(g), gq.get(g)) if isinstance(gq, dict) else None
                out.append((text or default, idx))
        return out
    return [(default, sorted(legal))]
