from types import SimpleNamespace

import torch

from src.reader.qwen_reader import QwenReader


def test_raw_first_logits_are_reused_not_last_logits_or_forced_answer_scores():
    class Model(torch.nn.Module):
        def forward(self, step):
            return SimpleNamespace(logits=torch.tensor([[[0., 1., 2.]]] if step == 0 else [[[20., 0., 0.]]]))

        def generate(self, input_ids, **kw):
            assert kw['do_sample'] is False
            self(0)
            self(1)
            return torch.cat([input_ids, torch.tensor([[2, 0]])], dim=1)

    reader = QwenReader.__new__(QwenReader)
    reader.model, reader.device, reader.sdpa_backend = Model(), torch.device('cpu'), None
    reader.cfg = {'attn_implementation': 'sdpa'}
    reader.processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0), batch_decode=lambda *a, **kw: ['C'])
    reader._prepare = lambda messages: ({'input_ids': torch.tensor([[0, 1]])}, [2], [0])
    reader._letter_id = lambda x: 'ABC'.index(x)
    reader.stats = dict(calls=0, seconds=0., prompt_tokens=0, visual_tokens=0, new_tokens=0)
    texts, meta = reader._generate([[]], 8, [list('ABC')])
    assert texts == ['C']
    torch.testing.assert_close(torch.tensor(list(meta[0]['letter_probs'].values())), torch.tensor([0., 1., 2.]).softmax(-1))
    assert not reader.model._forward_hooks  # hooks never leak into subsequent requests
    assert reader.stats['calls'] == 1
