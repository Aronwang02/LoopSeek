"""CPU orchestration checks only: no models, image preprocessing or engine comparison."""
import pytest

from src.reader.qwen_reader import QwenReader
from src.reader.vllm_reader import VLLMReader


@pytest.fixture
def reader(monkeypatch):
    reader = VLLMReader.__new__(VLLMReader)
    reader.cfg = dict(batch_size=2, observation_max_new_tokens=16, answer_max_new_tokens=8,
                      answer_fallback='letter_logits')
    calls = []

    def generate(messages, max_new_tokens, letters_batch=None):
        calls.append((list(messages), max_new_tokens))
        # Exercise generated answers and the first-token fallback without another call.
        return ([f'observation {m}' if letters_batch is None else ('A' if m % 2 == 0 else '')
                 for m in messages],
                [dict(question=m, letter_probs={'A': .2, 'B': .8}) for m in messages])

    reader._generate = generate
    monkeypatch.setattr('src.reader.qwen_reader.answer_messages', lambda frames, times, question, *args: question)
    monkeypatch.setattr('src.reader.qwen_reader.observe_messages', lambda frames, times, question, *args: question)
    return reader, calls


def jobs():
    return [dict(kind='answer', question=i, frames=[], times=[], options={'A': 'a', 'B': 'b'},
                 notes=[], duration=60, source='', interval=[0, 60]) for i in range(9)]


def test_answer_queue_is_not_split_and_order_and_fallback_are_preserved(reader):
    r, calls = reader
    out = r.run(jobs())
    assert calls == [(list(range(9)), 8)]
    assert [x['meta']['question'] for x in out] == list(range(9))
    assert [x['parsed'] for x in out] == ['A' if i % 2 == 0 else 'B' for i in range(9)]
    assert out[1]['meta']['answer_source'] == 'letter_logits_fallback'


def test_generic_and_label_requests_are_each_submitted_once(reader):
    r, calls = reader
    texts, meta = r.generate(list(range(9)), 23)
    assert len(texts) == len(meta) == 9
    assert calls == [(list(range(9)), 23)]
    calls.clear()
    assert r.answer_letter_probs(jobs()) == [{'A': .2, 'B': .8}] * 9
    assert calls == [(list(range(9)), 1)]


def test_mixed_kinds_restore_original_order_and_empty_requests_do_not_call_engine(reader):
    r, calls = reader
    mixed = jobs()
    for i in (1, 4, 7):
        mixed[i]['kind'] = 'observe'
    out = r.run(mixed)
    assert calls == [([1, 4, 7], 16), ([0, 2, 3, 5, 6, 8], 8)]
    assert [x['meta']['question'] for x in out] == list(range(9))
    calls.clear()
    assert r.run([]) == r.answer_letter_probs([]) == []
    assert r.generate([], 8) == ([], [])
    assert calls == []


def test_hf_still_obeys_its_fixed_batch_memory_limit():
    r = QwenReader.__new__(QwenReader)
    r.cfg = dict(batch_size=2)
    calls = []

    def generate(chunk, max_new_tokens):
        calls.append(list(chunk))
        return list(chunk), [{} for _ in chunk]

    r._generate = generate
    texts, _ = r.generate(list(range(5)), 8)
    assert calls == [[0, 1], [2, 3], [4]]
    assert texts == list(range(5))
