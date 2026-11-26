"""Multimodal decision policy: native OneJev letters or legacy NanoJev candidate scoring.

The OneJev route reads pretrained option-letter embedding rows, uses frozen native image tokens and role-specific
LoRA adapters, and does not use the legacy head/projector below. Single questions run as complete prompts; grouped
questions and opt-in static inference caches can share a prefix. Training retains full-prefix gradients.

Legacy NanoJev route:

* backbone: NanoJev's Qwen3-0.6B (unified-games-v1 weights), frozen, with LoRA (r=16, alpha=32);
  policy.finetune='full' instead trains every backbone weight (no LoRA);
* projector: Linear(d_visual, d_hidden) -> GELU -> Linear(d_hidden, d_hidden), trainable;
* one Choice head (NanoJev's: LayerNorm -> scalar + set-attention residual), shared by the
  controller, localizer and frame selector; the task is given in the input text.

Scoring follows NanoJev's DecisionModel (third_party/NanoJev/scripts/train_toy_decisions.py):
each legal candidate is its own path  [prefix][candidate]<eos>; the hidden state at the last valid
position feeds the head; a masked softmax over legal candidates gives the distribution (illegal
candidates have probability exactly 0). Each frame reference (VisRef) becomes `tokens_per_frame`
projected SigLIP2 tokens inserted through inputs_embeds. The prefix is repeated per candidate
(no prefix sharing), as in NanoJev.
"""
import collections
import contextlib
import re
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from .lora import apply_lora

HEAD_KEYS = ("norm", "scalar", "set_project", "set_attention", "set_output")

# policy.strip_vestigial (2026-09-30, design B, agreed with Astra and the user): the frames-only pipeline has no reader
# observations and no inspections, so these state lines of the old multi-turn pipeline are constants; they are removed
# from the policy's text on the fly (stored decision inputs unchanged). "- Reader observations:" spans its indented
# continuation lines.
_VESTIGIAL = re.compile(r"- Frames sent to the reader: [^\n]*\n|- Reader observations:\n(?:  [^\n]*\n)*|- Inspections left: [^\n]*\n")


def strip_vestigial(text):
    return _VESTIGIAL.sub("", text)


class DecisionHead(nn.Module):
    """NanoJev's Choice head parameterization (norm, scalar, set attention)."""

    def __init__(self, hidden, set_head=True):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.scalar = nn.Linear(hidden, 1)
        nn.init.normal_(self.scalar.weight, std=0.02)
        nn.init.zeros_(self.scalar.bias)
        self.set_head = set_head
        if set_head:
            self.set_project = nn.Linear(hidden + 1, 128)
            self.set_attention = nn.MultiheadAttention(128, 4, dropout=0.0, batch_first=True)
            self.set_output = nn.Linear(128, 1)
            nn.init.zeros_(self.set_output.weight)
            nn.init.zeros_(self.set_output.bias)

    def forward(self, h, valid):
        """h: [B, K, H] last-position states; valid: [B, K] candidate present. Returns fp32 [B, K]."""
        h = self.norm(h)
        z = self.scalar(h).squeeze(-1).float()
        if self.set_head:
            log_k = valid.sum(-1).clamp(min=1).float().log()[:, None, None].expand(-1, h.shape[1], 1)
            u = self.set_project(torch.cat([h, log_k.to(h.dtype)], dim=-1))
            mixed, _ = self.set_attention(u, u, u, key_padding_mask=~valid, need_weights=False)
            z = z + self.set_output(torch.tanh(u + mixed)).squeeze(-1).float()
        return z


class Projector(nn.Module):
    """Linear(d_visual, d_hidden) -> GELU -> Linear(d_hidden, d_hidden) (update.md §3).
    fc2 is rescaled once at init so that unit-variance inputs give outputs with the RMS of the
    backbone's text embeddings (visual tokens start on the text-embedding scale)."""

    def __init__(self, vis_dim, hidden, target_rms=None):
        super().__init__()
        self.fc1 = nn.Linear(vis_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        if target_rms is not None:
            with torch.no_grad():
                g = torch.Generator().manual_seed(0)
                x = torch.randn(512, vis_dim, generator=g)
                y = self.fc2(F.gelu(self.fc1(x)))
                s = target_rms / y.pow(2).mean().sqrt().item()
                self.fc2.weight.mul_(s)
                self.fc2.bias.mul_(s)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class PriorScorer:
    """Frozen SigLIP2 text tower + image attention-pooling head. prior(candidate) = logit_scale *
    max over (query text, candidate frame) of cos(text emb, head(stored 2x2 tokens)); the same score as
    scripts/siglip_prior_check.py ("maxopt": question + each option). Not an nn.Module on purpose:
    it holds no trainable weight and is shared by the policy and its frozen reference copy."""

    def __init__(self, model_path, device="cuda"):
        from transformers import AutoModel, AutoTokenizer
        m = AutoModel.from_pretrained(model_path, dtype=torch.float32).to(device).eval()
        m.requires_grad_(False)
        self.text_model, self.head = m, m.vision_model.head
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.scale = float(m.logit_scale.exp())
        self.device = torch.device(device)
        self._text, self._img = OrderedDict(), OrderedDict()

    @staticmethod
    def _t(x):
        return x if torch.is_tensor(x) else x.pooler_output

    @torch.no_grad()
    def text_emb(self, texts):
        miss = [t for t in texts if t not in self._text]
        if miss:
            b = self.tok(miss, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            e = F.normalize(self._t(self.text_model.get_text_features(input_ids=b["input_ids"].to(self.device))).float(), dim=-1)
            for t, v in zip(miss, e):
                self._text[t] = v
            while len(self._text) > 50_000:
                self._text.popitem(last=False)
        return torch.stack([self._text[t] for t in texts])

    @torch.no_grad()
    def img_emb(self, refs, store):
        keys = [(r.content_id, int(r.pts)) for r in refs]
        miss = list(dict.fromkeys(k for k in keys if k not in self._img))
        if miss:
            g = store.get(miss).to(self.device).float()
            if g.dim() == 2:
                g = g[:, None, :]
            e = F.normalize(self._t(self.head(g)).float(), dim=-1)
            for k, v in zip(miss, e):
                self._img[k] = v
            while len(self._img) > 200_000:
                self._img.popitem(last=False)
        return torch.stack([self._img[k] for k in keys])

    @torch.no_grad()
    def scores(self, decision, store):
        """[K] prior logits (0 for candidates without frames), K = len(candidates). Always fp32: the policy
        forward runs under bf16 autocast, and cosines are multiplied by logit_scale (~117)."""
        with torch.autocast(self.device.type, enabled=False):
            return self._scores(decision, store)

    def _scores(self, decision, store):
        k = len(decision.candidates)
        out = torch.zeros(k, device=self.device)
        if not decision.prior_texts or decision.prior_refs is None:
            return out
        te = self.text_emb(decision.prior_texts)                          # [m, D]
        flat = [(i, r) for i, rs in enumerate(decision.prior_refs) for r in rs]
        if not flat:
            return out
        sim = (self.img_emb([r for _, r in flat], store) @ te.T).max(dim=1).values   # [n]
        idx = torch.tensor([i for i, _ in flat], device=self.device)
        best = torch.full((k,), float("-inf"), device=self.device).scatter_reduce(0, idx, sim, "amax")
        return torch.where(torch.isfinite(best), best * self.scale, out)


class PriorGate(nn.Module):
    """Learnable weight on the SigLIP2 prior. policy.prior_gate_init (2026-09-30: calibrated so that the prior logits
    have unit std across the candidates of a decision; 1.0 = SigLIP's own logit scale)."""

    def __init__(self, init=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.full((), float(init)))


class JevPolicy(nn.Module):
    def __init__(self, backbone, tokenizer, vis_dim=768, tokens_per_frame=4, set_head=True):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        hidden = backbone.config.hidden_size
        with torch.no_grad():
            emb = backbone.get_input_embeddings().weight
            self.text_emb_rms = emb.float().pow(2).mean(-1).sqrt().mean().item()
        self.projector = Projector(vis_dim, hidden, target_rms=self.text_emb_rms)
        self.head = DecisionHead(hidden, set_head)
        self.tokens_per_frame = tokens_per_frame
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self._tok_cache = OrderedDict()
        self.feature_store = None   # set by the caller; resolves VisRef -> frozen feature [T, D]
        self.token_store = None     # optional separate store for the frame TOKENS (frame_tokens=global1)
        self.lora_modules = []
        self.finetune = "lora"       # 'lora' | 'full', set by configure_finetune
        self.visual_mode = "normal"  # 'zero': text-only control, visual embeddings zeroed after the projector
        self.candidate_vis = False   # 2026-09-30 exp. 1: frame candidates also carry their own frame's tokens
        self.prior_scorer = None     # PriorScorer (policy.siglip_prior); set by build_policy, not a submodule
        self.prior_gate = None       # PriorGate when the prior is enabled
        # frame_tokens=qwen_native (2026-09-30, design B): frames are Qwen3.5 image tokens from its own frozen tower +
        # merger (no projector); each VisRef = <|vision_start|> + grid^2 image tokens + <|vision_end|>, and the backbone
        # gets Qwen3.5's M-RoPE positions (text: t=h=w=p; image: t=p, h=p+row, w=p+col, then p += grid)
        self.native = False
        self.vision_start_id = self.vision_end_id = None
        self.strip_vestigial = False
        # policy.prefix_sharing (2026-10-01, design B): without gradients (rollouts, evaluation, audits, reference
        # logits) each decision's prefix is run once and its cache (full-attention KV + Gated DeltaNet conv/recurrent
        # states) is copied to every candidate, which then runs only its own suffix; with gradients the full paths run
        self.prefix_sharing = False
        # policy.prefix_sharing_train (2026-10-01): the same with gradients (SFT, RL updates); the gradient of every
        # candidate flows back through the copied cache into the one prefix forward (out-of-place cache, no layer
        # checkpointing during that forward, see forward_shared)
        self.prefix_sharing_train = False
        # policy.static_prefix_cache (2026-10-06): reuse the static head (task / question / options / overview) across
        # decisions of one role in no-grad forwards; static_cache_max snapshots (LRU), stats for logging
        self.static_cache = False
        self.static_cache_max = 32
        # policy.readout (2026-10-06): head = DecisionHead on one path per candidate; letters = OneJev letter readout
        self.readout = "head"
        self._chat = self._slots = None
        self.forward_max_paths = 64
        self.forward_max_tokens = 98304
        self.static_cache_stats = {"hit": 0, "miss": 0}
        # inference speedups A1-A4 (all no-grad only, training paths unchanged):
        # policy.one_pass_single (A2): a single-question decision with a static snapshot runs the rest of its prefix and
        #   its question suffix as ONE forward (the cache after it is discarded), not a prefix-rest forward + a suffix one;
        # policy.suffix_batch_max (A4): at most this many question rows per suffix forward, each sub-batch on its own copy
        #   of the prefix-cache rows (a 96-question overview decision materialized 96 copies of an ~8k-token KV, ~25 GB,
        #   and ran out of memory next to the vLLM reader); 0 = all rows at once (the old behaviour);
        # policy.merge_lora_inference (A1): eval-mode no-grad forwards use LoRA folded into a bf16 copy of each base
        #   weight (src/policy/lora.py ensure_merged; one buffer per base weight shared by the roles, re-merged when
        #   another role runs), so a forward launches ~1.5k fewer small kernels.
        self.one_pass_single = False
        self.suffix_batch_max = 0
        self.merge_lora = False
        self._merge_version = 0
        # policy.sort_suffixes (V1) / policy.shared_prefix_attn (V2), 2026-11-03, no-grad only: question suffixes sorted by
        # length before sub-batching (no padding inside a sub-batch); suffixes read the prefix KV in place
        self.sort_suffixes = False
        self.shared_prefix_attn = False

    def enable_prior(self, scorer, gate_init=1.0):
        self.prior_scorer = scorer
        if self.prior_gate is None:
            self.prior_gate = PriorGate(gate_init).to(self.projector.fc1.weight.device)

    # ---------------------------------------------------------------- params
    def param_groups(self):
        """(LoRA params, new params = projector + head); frozen backbone weights excluded."""
        lora = [p for n, p in self.named_parameters() if "lora_" in n and p.requires_grad]
        new = [p for n, p in self.named_parameters()
               if (n.startswith("projector.") or n.startswith("head.") or n.startswith("prior_gate."))
               and p.requires_grad]
        return lora, new

    def backbone_params(self):
        """Trained backbone base weights (finetune='full'); empty under LoRA."""
        return [p for n, p in self.named_parameters()
                if n.startswith("backbone.") and "lora_" not in n and p.requires_grad]

    def trainable_names(self):
        return [n for n, p in self.named_parameters() if p.requires_grad]

    def trainable_state_dict(self):
        names = set(self.trainable_names())
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k in names}

    def load_trainable_state_dict(self, sd):
        names = set(self.trainable_names())
        missing, unexpected = names - set(sd), set(sd) - names
        if missing or unexpected:
            raise KeyError(f"trainable state mismatch: missing={sorted(missing)[:5]} "
                           f"unexpected={sorted(unexpected)[:5]}")
        self.load_state_dict(sd, strict=False)
        self.clear_static_cache()

    def train(self, mode=True):
        self._merge_version = getattr(self, "_merge_version", 0) + 1   # merged LoRA weights are rebuilt after any switch
        if "backbone" in self._modules:
            from .lora import release_merged
            release_merged(self.backbone)
        if mode:
            self.clear_static_cache()       # no inference state may survive an optimizer update
        return super().train(mode)

    # ---------------------------------------------------------------- text
    def encode_text(self, text):
        ids = self._tok_cache.get(text)
        if ids is None:
            ids = self.tokenizer.encode(strip_vestigial(text) if self.strip_vestigial else text, add_special_tokens=False)
            self._tok_cache[text] = ids
            if len(self._tok_cache) > 200_000:
                self._tok_cache.popitem(last=False)
        return ids

    def text_len(self, text):
        return len(self.encode_text(text))

    def _segments_to_ids(self, segs, vis_items):
        """ids with -1 at visual positions (tokens_per_frame per VisRef); VisRefs appended in order."""
        ids = []
        for s in segs:
            if isinstance(s, str):
                ids.extend(self.encode_text(s))
            elif self.native:
                ids.extend([self.vision_start_id] + [-1] * self.tokens_per_frame + [self.vision_end_id])
                vis_items.append(s)
            else:
                ids.extend([-1] * self.tokens_per_frame)
                vis_items.append(s)
        return ids

    def _mrope_positions(self, ids):
        """[3, L] Qwen3.5 M-RoPE positions (Qwen3_5Model.get_rope_index) for one path; runs of -1 are image blocks of
        tokens_per_frame tokens each, row-major over a grid x grid merged layout."""
        g = int(round(self.tokens_per_frame ** 0.5))
        assert g * g == self.tokens_per_frame, self.tokens_per_frame
        blk = torch.arange(g * g)
        rows, cols = blk // g, blk % g
        out, p, i, L = [], 0, 0, len(ids)
        while i < L:
            if ids[i] >= 0:
                j = i
                while j < L and ids[j] >= 0:
                    j += 1
                out.append(torch.arange(p, p + j - i).expand(3, -1))
                p += j - i
                i = j
            else:
                blk_ids = ids[i:i + g * g]
                assert len(blk_ids) == g * g and all(x < 0 for x in blk_ids), "image block shorter than tokens_per_frame"
                assert i + g * g == L or ids[i + g * g] >= 0, "image block longer than tokens_per_frame"
                out.append(torch.stack([torch.full((g * g,), p), p + rows, p + cols]))
                p += g
                i += g * g
        return torch.cat(out, 1)

    @staticmethod
    def _next_position(pos):
        """The M-RoPE position of the first text token after a path whose positions are `pos` ([3, L]): the running
        counter of _mrope_positions after the path, i.e. max + 1 whether the path ends in text (p - 1) or in an image
        block (p - g + (g - 1) = p - 1 at the last row / column). Suffixes are text, so their positions continue from
        here (2026-11-03 A3: no per-question recomputation over the whole prefix)."""
        return int(pos.max()) + 1 if pos.numel() else 0

    def _letter_logits_into(self, logits, leaves, rows, weight, device):
        """Option-letter logits of many questions at once (2026-11-03 A3): leaves [n, D], one per question; rows
        [(decision row b, candidate indices of that question)]. The same per-element dot products as the former
        per-question GEMV (one fp32 GEMM instead), written with one scatter instead of one host->device copy each."""
        if not rows:
            return
        kq = max(len(idx) for _, idx in rows)
        with torch.autocast(device.type, enabled=False):
            z = leaves.float() @ weight[torch.tensor(self._slots[:kq], device=device)].float().T    # [n, kq]
        rb, rc, ri, rj = [], [], [], []
        for i, (b, idx) in enumerate(rows):
            rb += [b] * len(idx)
            rc += list(idx)
            ri += [i] * len(idx)
            rj += range(len(idx))
        t = lambda v: torch.tensor(v, dtype=torch.long, device=device)
        logits[t(rb), t(rc)] = z[t(ri), t(rj)]

    _CAND_ID = re.compile(r"(?:^|\n)(C\d+) \[[^\]]*\] $")

    def _candidates(self, d):
        """Candidate segment lists. With candidate_vis (frame decisions only) each candidate "Ck: frame Ck at ts"
        gets the VisRef that follows "Ck [ts] " in the prefix, inserted before its "\nDecision:" line, so the decision
        position attends to its own frame directly (the prefix copy stays). Stored DecisionInputs are unchanged."""
        if not self.candidate_vis or d.decision_type != "frame":
            return d.candidates
        own = {}
        for a, b in zip(d.prefix, d.prefix[1:]):
            if isinstance(a, str) and not isinstance(b, str):
                m = self._CAND_ID.search(a)
                if m:
                    own[m.group(1)] = b
        out = []
        for cid, c in zip(d.candidate_ids, d.candidates):
            ref = own.get(cid)
            if ref is None or len(c) != 1 or not isinstance(c[0], str) or not c[0].endswith("\nDecision:"):
                out.append(c)
                continue
            out.append([c[0][:-len("\nDecision:")] + " ", ref, "\nDecision:"])
        return out

    def path_lengths(self, decision):
        """Full prompt length per letter question, or per legal candidate for the legacy head readout."""
        if self.readout == "letters":
            spec = self._letter_spec(decision)
            return [len(spec["prefix"]) + len(s) for s in spec["sufs"]]
        prefix = self._segments_to_ids(decision.prefix, [])
        cands = self._candidates(decision)
        return [len(prefix) + len(self._segments_to_ids(cands[k], [])) + 1
                for k in decision.legal_indices()]

    # ---------------------------------------------------------------- forward
    def build_paths(self, decisions):
        paths, path_vis, owners = [], [], []
        stats = {"paths": 0, "text_tokens": 0, "vis_tokens": 0, "max_path": 0}
        for b, d in enumerate(decisions):
            pv = []
            prefix = self._segments_to_ids(d.prefix, pv)
            cands = self._candidates(d)
            for k in d.legal_indices():
                cv = []
                cand = self._segments_to_ids(cands[k], cv)
                ids = prefix + cand + [self.eos_id]
                paths.append(ids)
                path_vis.append(pv + cv)
                owners.append((b, k))
                n_vis = (len(pv) + len(cv)) * self.tokens_per_frame
                stats["text_tokens"] += len(ids) - n_vis
                stats["vis_tokens"] += n_vis
                stats["max_path"] = max(stats["max_path"], len(ids))
        stats["paths"] = len(paths)
        return paths, path_vis, owners, stats

    def forward(self, decisions):
        """Returns (logits [B, Kmax] fp32, -inf at illegal/absent candidates, stats)."""
        if self.merge_lora and not self.training and not torch.is_grad_enabled():
            from .lora import ensure_merged
            ensure_merged(self.backbone, (id(self), self._merge_version))
        if self.readout == "letters":
            return self._forward_letters(decisions)
        if self.prefix_sharing and (self.prefix_sharing_train or not torch.is_grad_enabled()):
            return self.forward_shared(decisions)
        device = self.projector.fc1.weight.device
        paths, path_vis, owners, stats = self.build_paths(decisions)
        kmax = max(len(d.candidates) for d in decisions)
        logits = torch.full((len(decisions), kmax), float("-inf"), device=device)
        n = len(paths)
        if n == 0:
            return logits, stats
        lengths = torch.tensor([len(p) for p in paths], device=device)
        width = int(lengths.max())
        ids = torch.full((n, width), self.pad_id, dtype=torch.long, device=device)
        vis_pos = torch.zeros((n, width), dtype=torch.bool, device=device)
        for i, p in enumerate(paths):
            t = torch.tensor(p, device=device)
            vis_pos[i, :len(p)] = t < 0
            ids[i, :len(p)] = torch.where(t < 0, torch.full_like(t, self.pad_id), t)
        embeds = self.backbone.get_input_embeddings()(ids)
        refs = [r for pv in path_vis for r in pv]
        if refs:
            vis = self._project(refs, device)                       # [n_refs * T, H], path order
            embeds = embeds.masked_scatter(vis_pos[..., None].expand_as(embeds), vis.to(embeds.dtype))
        attn = torch.arange(width, device=device)[None, :] < lengths[:, None]
        kw = {}
        if self.native:
            pos = torch.zeros((3, n, width), dtype=torch.long)
            for i, p in enumerate(paths):
                pos[:, i, :len(p)] = self._mrope_positions(p)
            kw["position_ids"] = pos.to(device)
        hidden = self.backbone(inputs_embeds=embeds, attention_mask=attn, use_cache=False, **kw).last_hidden_state
        leaves = hidden[torch.arange(n, device=device), lengths - 1]
        bs = sorted({b for b, _ in owners})
        slot = {b: i for i, b in enumerate(bs)}
        h = leaves.new_zeros((len(bs), kmax, leaves.shape[-1]))
        present = torch.zeros((len(bs), kmax), dtype=torch.bool, device=device)
        for j, (b, k) in enumerate(owners):
            h[slot[b], k] = leaves[j]
            present[slot[b], k] = True
        z = self.head(h, present)
        if self.prior_gate is not None and self.prior_scorer is not None:
            pri = torch.zeros_like(z)
            for b in bs:
                d = decisions[b]
                if d.prior_texts:
                    pri[slot[b], :len(d.candidates)] = self.prior_scorer.scores(d, self.feature_store)
            z = z + self.prior_gate.weight.float() * pri
        for b in bs:
            row, m = z[slot[b]], present[slot[b]]
            logits[b] = torch.where(m, row, torch.full_like(row, float("-inf")))
        return logits, stats

    # ---------------------------------------------------------------- prefix sharing (no grad)
    @staticmethod
    def _repeat_cache(cache, n):
        """Repeat cache rows n times (scalar or per-row counts), including KV, conv and recurrent state."""
        for layer in cache.layers:
            for name in ("keys", "values"):
                t = getattr(layer, name, None)
                if torch.is_tensor(t) and t.numel() > 0:
                    setattr(layer, name, t.repeat_interleave(n, dim=0))
            for name in ("conv_states", "recurrent_states"):      # dict {state_idx: tensor} in transformers 5.x
                st = getattr(layer, name, None)
                if isinstance(st, (dict, list)):
                    for i in (list(st.keys()) if isinstance(st, dict) else range(len(st))):
                        if torch.is_tensor(st[i]):
                            st[i] = st[i].repeat_interleave(n, dim=0)

    def _embed(self, ids, vis_refs, device):
        t = torch.tensor(ids, device=device)
        vis_pos = t < 0
        emb = self.backbone.get_input_embeddings()(torch.where(vis_pos, torch.full_like(t, self.pad_id), t))
        if vis_refs:
            vis = self._project(vis_refs, device)
            emb = emb.masked_scatter(vis_pos[:, None].expand_as(emb), vis.to(emb.dtype))
        return emb

    def shared_tokens(self, decision):
        """Prefix once + all question suffixes (letter readout) or legal candidate suffixes (legacy head)."""
        if self.readout == "letters":
            spec = self._letter_spec(decision)
            return len(spec["prefix"]) + sum(map(len, spec["sufs"]))
        legal = decision.legal_indices()
        if not legal:
            return 0
        return len(self._segments_to_ids(decision.prefix, [])) + sum(
            len(self._segments_to_ids(decision.candidates[k], [])) + 1 for k in legal)

    @contextlib.contextmanager
    def _no_layer_checkpointing(self):
        """Layers with gradient checkpointing on silently drop past_key_values whenever they are in train mode, with or
        without gradients (transformers GradientCheckpointingLayer checks only `gradient_checkpointing and training`),
        which would give every suffix an empty prefix (e.g. train_sft's dev evaluation runs under no_grad in train
        mode); checkpointing is off for every shared forward and restored after."""
        on = [m for m in self.backbone.modules() if getattr(m, "gradient_checkpointing", False) is True]
        for m in on:
            m.gradient_checkpointing = False
        try:
            yield
        finally:
            for m in on:
                m.gradient_checkpointing = True

    def _layer_checkpointing_on(self):
        return any(getattr(m, "gradient_checkpointing", False) is True for m in self.backbone.modules())

    @contextlib.contextmanager
    def _checkpointed_prefill(self, enabled):
        """Checkpoint every decoder layer of a forward that fills an EMPTY cache with gradients (the shared prefix of
        grouped training decisions, inside _no_layer_checkpointing). Each layer runs on a fresh cache and its new state
        then replaces that layer's entry in the real cache, so the recomputation during backward never writes the real
        cache, which is why transformers drops caches under checkpointing. Without it the prefix kept every layer's
        activations until backward (2026-10-12: a 128-frame verifier prefix of ~10k tokens ran out of memory on an
        80 GB A100). Logits and gradients are unchanged; backward recomputes the prefix one layer at a time."""
        if not enabled:
            yield
            return
        from torch.utils.checkpoint import checkpoint
        from transformers import DynamicCache
        layers = list(self.backbone.layers)
        own = [layer.__dict__.get("forward") for layer in layers]

        def wrap(i, forward):
            def run(hidden_states, *args, past_key_values=None, **kwargs):
                if past_key_values is None:
                    return forward(hidden_states, *args, **kwargs)
                first = []

                def on_fresh_cache(h):
                    fresh = out_of_place_linear_cache(DynamicCache(config=self.backbone.config))
                    if not first:                # the recomputation in backward writes a throwaway cache
                        first.append(fresh)
                    return forward(h, *args, past_key_values=fresh, **kwargs)

                out = checkpoint(on_fresh_cache, hidden_states, use_reentrant=False)
                past_key_values.layers[i] = first[0].layers[i]
                return out
            return run

        for i, layer in enumerate(layers):
            layer.forward = wrap(i, layer.forward)
        try:
            yield
        finally:
            for layer, forward in zip(layers, own):
                if forward is None:
                    del layer.forward
                else:
                    layer.forward = forward

    def forward_shared(self, decisions):
        """Same outputs as forward() (up to kernel numerics), computing each prefix once. Candidate segments must be
        text only (candidate_vis off); otherwise this falls back to the full paths. With gradients enabled the cache is
        updated out of place, so the candidates' gradients reach the prefix forward."""
        device = self.projector.fc1.weight.device
        if self.candidate_vis or any(not isinstance(x, str) for d in decisions for c in d.candidates for x in c):
            prev, self.prefix_sharing = self.prefix_sharing, False
            try:
                return self.forward(decisions)
            finally:
                self.prefix_sharing = prev
        from transformers import DynamicCache
        grad = torch.is_grad_enabled()
        with self._no_layer_checkpointing():
            return self._forward_shared(decisions, DynamicCache, grad)

    def _forward_shared(self, decisions, DynamicCache, grad):
        device = self.projector.fc1.weight.device
        kmax = max(len(d.candidates) for d in decisions)
        logits = torch.full((len(decisions), kmax), float("-inf"), device=device)
        stats = {"paths": 0, "text_tokens": 0, "vis_tokens": 0, "max_path": 0, "prefix_tokens": 0, "suffix_tokens": 0}
        dt = self.backbone.get_input_embeddings().weight.dtype
        for b, d in enumerate(decisions):
            legal = d.legal_indices()
            if not legal:
                continue
            pv = []
            prefix = self._segments_to_ids(d.prefix, pv)
            sufs = [self._segments_to_ids(d.candidates[k], []) + [self.eos_id] for k in legal]
            P, n = len(prefix), len(sufs)
            pos_full = [self._mrope_positions(prefix + s) if self.native else None for s in sufs]
            ns = int(getattr(d, "static_segments", 0) or 0)
            if self.static_cache and not grad and ns > 0:
                cache = self._static_then_dynamic(d, ns, prefix, pv, pos_full[0], P, device, dt, DynamicCache)
            else:
                emb = self._embed(prefix, pv, device)[None].to(dt) if pv else self._embed(prefix, [], device)[None]
                kw = {"position_ids": pos_full[0][:, None, :P].to(device)} if self.native else {}
                cache = DynamicCache(config=self.backbone.config)
                if grad:
                    out_of_place_linear_cache(cache)
                self.backbone(inputs_embeds=emb, attention_mask=torch.ones(1, P, dtype=torch.long, device=device),
                              past_key_values=cache, use_cache=True, **kw)
            self._repeat_cache(cache, n)
            lens = torch.tensor([len(s) for s in sufs], device=device)
            S = int(lens.max())
            ids = torch.full((n, S), self.pad_id, dtype=torch.long, device=device)
            for i, s in enumerate(sufs):
                ids[i, :len(s)] = torch.tensor(s, device=device)
            emb_s = self.backbone.get_input_embeddings()(ids)
            attn = torch.cat([torch.ones(n, P, dtype=torch.long, device=device),
                              (torch.arange(S, device=device)[None, :] < lens[:, None]).long()], 1)
            kw = {}
            if self.native:
                pos = torch.zeros((3, n, S), dtype=torch.long)
                for i, s in enumerate(sufs):
                    pos[:, i, :len(s)] = pos_full[i][:, P:]
                kw["position_ids"] = pos.to(device)
            else:
                kw["position_ids"] = (P + torch.arange(S, device=device))[None, :].expand(n, -1)
            h = self.backbone(inputs_embeds=emb_s, attention_mask=attn, past_key_values=cache, use_cache=True,
                              **kw).last_hidden_state
            leaves = h[torch.arange(n, device=device), lens - 1]
            hb = leaves.new_zeros((1, kmax, leaves.shape[-1]))
            present = torch.zeros((1, kmax), dtype=torch.bool, device=device)
            for j, k in enumerate(legal):
                hb[0, k] = leaves[j]
                present[0, k] = True
            z = self.head(hb, present)
            if self.prior_gate is not None and self.prior_scorer is not None and d.prior_texts:
                pri = torch.zeros_like(z)
                pri[0, :len(d.candidates)] = self.prior_scorer.scores(d, self.feature_store)
                z = z + self.prior_gate.weight.float() * pri
            logits[b] = torch.where(present[0], z[0], torch.full_like(z[0], float("-inf")))
            n_vis = len(pv) * self.tokens_per_frame
            stats["paths"] += n
            stats["prefix_tokens"] += P
            stats["suffix_tokens"] += int(lens.sum())
            stats["vis_tokens"] += n_vis
            stats["text_tokens"] += P - n_vis + int(lens.sum())
            stats["max_path"] = max(stats["max_path"], P + S)
        return logits, stats

    # ---------------------------------------------------------------- OneJev letter readout (2026-10-06)
    def _forward_letters(self, decisions):
        # Identical states in a rollout group need one scoring. Expand logits
        # before sampling so each trajectory keeps its own original RNG stream.
        if not torch.is_grad_enabled() and getattr(self, "deduplicate_inference", False) and len(decisions) > 1:
            unique, owners, seen = [], [], {}
            for d in decisions:
                s = self._letter_spec(d)
                key = (tuple(s["prefix"]), tuple((r.content_id, r.pts) for r in s["pv"]),
                       tuple(map(tuple, s["sufs"])), tuple(tuple(q[1]) for q in s["qs"]),
                       s["ns"], len(d.candidates))
                if key not in seen:
                    seen[key] = len(unique)
                    unique.append(d)
                owners.append(seen[key])
            if len(unique) < len(decisions):
                logits, stats = self._forward_letters_impl(unique)
                stats["deduplicated_decisions"] = len(decisions) - len(unique)
                return logits[torch.tensor(owners, device=logits.device)], stats
        return self._forward_letters_impl(decisions)

    def _forward_letters_impl(self, decisions):
        """Single questions use a full prompt; grouped questions can share their prefix.

        Splitting one question saves no repeated computation and changes BF16 chunk boundaries, which can materially
        change gradients on real frame/controller inputs. Full prompts also retain layer activation checkpointing.
        prefix_sharing_train=False requests full prompts for grouped training questions too. Inference uses the cached
        path when static-prefix reuse is enabled for a decision; otherwise a single question uses one complete prompt.
        """
        from .letters import questions_of
        grad = torch.is_grad_enabled()
        full_indices = [i for i, d in enumerate(decisions)
                        if (grad and not self.prefix_sharing_train) or
                        (len(questions_of(d)) <= 1 and
                         (grad or not (self.static_cache and int(getattr(d, "static_segments", 0) or 0) > 0)))]
        if not full_indices:
            return self._forward_letters_shared(decisions)
        if len(full_indices) == len(decisions):
            return self._forward_letters_full(decisions)
        shared_indices = [i for i in range(len(decisions)) if i not in full_indices]
        kmax = max(len(d.candidates) for d in decisions)
        logits = torch.full((len(decisions), kmax), -torch.inf, device=self.projector.fc1.weight.device)
        total = {}
        for indices, fn in ((full_indices, self._forward_letters_full),
                            (shared_indices, self._forward_letters_shared)):
            z, stats = fn([decisions[i] for i in indices])
            logits[indices, :z.shape[1]] = z
            for key, value in stats.items():
                total[key] = max(total.get(key, 0), value) if key == "max_path" else total.get(key, 0) + value
        return logits, total

    def _letter_spec(self, d):
        """Token layout shared by execution and the batch planner (no feature reads)."""
        from ..env.environment import merge_text
        from .letters import chat_parts, option_name_text, question_suffix, questions_of, slot_token_ids
        if self._chat is None:
            self._chat, self._slots = chat_parts(self.tokenizer), slot_token_ids(self.tokenizer)
        head, tail = self._chat
        ns = int(getattr(d, "static_segments", 0) or 0)
        if ns > 0 and not isinstance(d.prefix[ns - 1], str):
            head_segs = merge_text([head] + list(d.prefix[:ns]))
            segs = head_segs + merge_text(list(d.prefix[ns:-1]) + ["\n</state>\n\n"])
            ns = len(head_segs)
        else:
            segs, ns = merge_text([head] + list(d.prefix[:-1]) + ["\n</state>\n\n"]), 0
        pv = []
        prefix = self._segments_to_ids(segs, pv)
        qs = questions_of(d)
        assert all(len(idx) <= len(self._slots) for _, idx in qs), "more options than answer slots"
        sufs = [self._segments_to_ids([question_suffix(instr, [option_name_text(d.candidates[i], d.candidate_ids[i])
                                                              for i in idx]) + tail], []) for instr, idx in qs]
        return dict(prefix=prefix, pv=pv, ns=ns, segs=segs, qs=qs, sufs=sufs)

    def _letter_chunks(self, specs):
        """Bound padded prompt / KV footprint, including all questions of a grouped decision.

        One oversized decision is still allowed, as in the original single-decision implementation.
        """
        chunk, count, width = [], 0, 0
        for spec in specs:
            n = len(spec["sufs"])
            size = len(spec["prefix"]) + max(map(len, spec["sufs"]), default=0)
            if chunk and (count + n > self.forward_max_paths or
                          (count + n) * max(width, size) > self.forward_max_tokens):
                yield chunk
                chunk, count, width = [], 0, 0
            chunk.append(spec)
            count, width = count + n, max(width, size)
        if chunk:
            yield chunk

    def _embed_letter_batch(self, paths, visuals, device):
        """Right-pad independent prompts; project / load their visual references in one operation."""
        width = max(map(len, paths))
        ids = torch.full((len(paths), width), self.pad_id, dtype=torch.long, device=device)
        vis_pos = torch.zeros_like(ids, dtype=torch.bool)
        for i, path in enumerate(paths):
            t = torch.tensor(path, device=device)
            vis_pos[i, :len(path)] = t < 0
            ids[i, :len(path)] = t.clamp_min(0).masked_fill(t < 0, self.pad_id)
        embeds = self.backbone.get_input_embeddings()(ids)
        refs = [r for vs in visuals for r in vs]
        if refs:
            vis = self._project(refs, device)
            embeds = embeds.masked_scatter(vis_pos[..., None].expand_as(embeds), vis.to(embeds.dtype))
        lengths = torch.tensor([len(p) for p in paths], device=device)
        mask = torch.arange(width, device=device)[None] < lengths[:, None]
        return embeds, mask, lengths

    @staticmethod
    def _map_letter_caches(caches, transform):
        """Copy cache metadata and transform every batch tensor, including hybrid linear-attention state.

        Static inference snapshots remain immutable. No padded tokens are ever fed into a reusable linear state.
        """
        import copy
        result = copy.copy(caches[0])
        result.layers = []
        for i, template in enumerate(caches[0].layers):
            layer = copy.copy(template)
            # The backbone mutates initialization flags as well as state dictionaries.
            for name, value in vars(template).items():
                if isinstance(value, (dict, list)):
                    setattr(layer, name, value.copy())
            for name in ("keys", "values"):
                value = getattr(template, name, None)
                if torch.is_tensor(value) and value.numel():
                    setattr(layer, name, transform([getattr(c.layers[i], name) for c in caches]))
            for name in ("conv_states", "recurrent_states"):
                states = getattr(template, name, None)
                if isinstance(states, (dict, list)):
                    for j in (states.keys() if isinstance(states, dict) else range(len(states))):
                        if torch.is_tensor(states[j]):
                            getattr(layer, name)[j] = transform([getattr(c.layers[i], name)[j] for c in caches])
            result.layers.append(layer)
        return result

    def _letter_cached_forward(self, paths, visuals, positions, cache, past_length=0):
        """Equal-length prefixes only: right-padding would change conv/recurrent state before the next suffix."""
        device = self.projector.fc1.weight.device
        assert len({len(p) for p in paths}) == 1
        embeds, _, _ = self._embed_letter_batch(paths, visuals, device)
        kw = {"position_ids": torch.stack(positions, dim=1).to(device)} if self.native else {}
        if not self.native:
            kw["position_ids"] = torch.arange(past_length, past_length + len(paths[0]), device=device)[None].expand(len(paths), -1)
        self.backbone(inputs_embeds=embeds,
                      attention_mask=torch.ones(len(paths), past_length + len(paths[0]), dtype=torch.long, device=device),
                      past_key_values=cache, use_cache=True, **kw)

    def _letter_static_snapshots(self, specs, cache_type):
        """Batch cold static heads by exact length; hot heads are looked up independently per question."""
        if not hasattr(self, "_static_kv"):
            self.clear_static_cache()
        snapshots, missing = {}, collections.defaultdict(list)
        for s in specs:
            pv = []
            ids = self._segments_to_ids(s["segs"][:s["ns"]], pv)
            n = len(ids)
            assert s["prefix"][:n] == ids and n < len(s["prefix"])
            key = (tuple(ids), tuple((r.content_id, r.pts) for r in pv))
            s.update(static_key=key, static_length=n, static_visuals=len(pv))
            if key in snapshots:
                self.static_cache_stats["hit"] += 1
            elif key in self._static_kv:
                snapshots[key] = self._static_kv[key]
                self._static_kv.move_to_end(key)
                self.static_cache_stats["hit"] += 1
            else:
                snapshots[key] = None
                missing[n].append((key, ids, pv, s["positions"][:, :n] if self.native else None))
                self.static_cache_stats["miss"] += 1
        for n, group in missing.items():
            size = max(1, min(self.forward_max_paths, self.forward_max_tokens // n))
            for start in range(0, len(group), size):
                chunk = group[start:start + size]
                cache = cache_type(config=self.backbone.config)
                self._letter_cached_forward([x[1] for x in chunk], [x[2] for x in chunk], [x[3] for x in chunk], cache)
                for i, (key, _, _, _) in enumerate(chunk):
                    snap = self._map_letter_caches([cache], lambda ts: ts[0][i:i + 1].clone())
                    snapshots[key] = self._static_kv[key] = snap
                    while len(self._static_kv) > self.static_cache_max:
                        self._static_kv.popitem(last=False)
        for s in specs:
            s["snapshot"] = snapshots[s["static_key"]]

    def _forward_letters_full(self, decisions):
        """Batched complete prompts, retaining gradients and layer checkpointing."""
        device = self.projector.fc1.weight.device
        weight = self.backbone.get_input_embeddings().weight
        logits = torch.full((len(decisions), max(len(d.candidates) for d in decisions)), -torch.inf, device=device)
        stats = dict(paths=0, text_tokens=0, vis_tokens=0, max_path=0, prefix_tokens=0, suffix_tokens=0)
        jobs = []
        for b, d in enumerate(decisions):
            spec = self._letter_spec(d)
            if self.native:        # A3 (2026-11-03): the prefix's M-RoPE positions once per decision, not per question
                spec["prefix_pos"] = self._mrope_positions(spec["prefix"])
                spec["next_pos"] = self._next_position(spec["prefix_pos"])
            for q, suffix in zip(spec["qs"], spec["sufs"]):
                jobs.append(dict(spec, b=b, qs=[q], sufs=[suffix]))
        jobs.sort(key=lambda s: len(s["prefix"]) + len(s["sufs"][0]))
        for chunk in self._letter_chunks(jobs):
            paths = [s["prefix"] + s["sufs"][0] for s in chunk]
            embeds, mask, lengths = self._embed_letter_batch(paths, [s["pv"] for s in chunk], device)
            kw = {}
            if self.native:
                pos = torch.zeros(3, len(paths), embeds.shape[1], dtype=torch.long)
                for i, s in enumerate(chunk):
                    P, n = len(s["prefix"]), len(s["sufs"][0])
                    pos[:, i, :P] = s["prefix_pos"]
                    pos[:, i, P:P + n] = s["next_pos"] + torch.arange(n)     # text-only suffix: t = h = w
                kw["position_ids"] = pos.to(device)
            h = self.backbone(inputs_embeds=embeds, attention_mask=mask, use_cache=False, **kw).last_hidden_state
            leaves = h[torch.arange(len(paths), device=device), lengths - 1]
            self._letter_logits_into(logits, leaves, [(s["b"], s["qs"][0][1]) for s in chunk], weight, device)
            for i, s in enumerate(chunk):
                visual = len(s["pv"]) * self.tokens_per_frame
                stats["paths"] += 1
                stats["prefix_tokens"] += len(s["prefix"])
                stats["suffix_tokens"] += len(s["sufs"][0])
                stats["text_tokens"] += len(paths[i]) - visual
                stats["vis_tokens"] += visual
                stats["max_path"] = max(stats["max_path"], len(paths[i]))
        return logits, stats

    def _forward_letters_shared(self, decisions):
        """Batch compatible prefix/cache states, then expand each state to its own question suffixes.

        Prefix lengths must match inside a batch so hybrid linear-attention state is never padded. Final suffixes
        may be right-padded because their resulting cache is discarded. Static snapshots are used only without grad.
        With gradients and layer checkpointing on, the prefix forward is checkpointed layer by layer
        (_checkpointed_prefill); the suffixes are short and keep their activations.
        """
        from transformers import DynamicCache
        device = self.projector.fc1.weight.device
        grad = torch.is_grad_enabled()
        weight = self.backbone.get_input_embeddings().weight
        logits = torch.full((len(decisions), max(len(d.candidates) for d in decisions)), -torch.inf, device=device)
        stats = dict(paths=0, text_tokens=0, vis_tokens=0, max_path=0, prefix_tokens=0, suffix_tokens=0)
        specs = [dict(self._letter_spec(d), b=b) for b, d in enumerate(decisions)]
        specs = [s for s in specs if s["sufs"]]
        for s in specs:           # A3 (2026-11-03): the prefix's positions once; text suffixes continue from next_pos
            s["positions"] = self._mrope_positions(s["prefix"]) if self.native else None
            s["next_pos"] = self._next_position(s["positions"]) if self.native else len(s["prefix"])
        checkpointed = grad and self._layer_checkpointing_on()
        with self._no_layer_checkpointing():
            cached = [s for s in specs if self.static_cache and not grad and s["ns"] > 0]
            if cached:
                self._letter_static_snapshots(cached, DynamicCache)
            one = ({id(s) for s in specs if s.get("static_length", 0) and len(s["sufs"]) == 1}
                   if self.one_pass_single and not grad else set())
            groups = collections.defaultdict(list)
            for s in specs:
                if id(s) in one:
                    self._letter_one_pass(s, logits, stats, weight, device)
                else:
                    groups[(len(s["prefix"]), s.get("static_length", 0))].append(s)
            for (P, Ps), group in groups.items():
                for chunk in self._letter_chunks(group):
                    if Ps:
                        cache = self._map_letter_caches([s["snapshot"] for s in chunk], lambda ts: torch.cat(ts, dim=0))
                    else:
                        cache = DynamicCache(config=self.backbone.config)
                        if grad:
                            out_of_place_linear_cache(cache)
                    with self._checkpointed_prefill(checkpointed and not Ps):   # Ps: a snapshot, never with grad
                        self._letter_cached_forward([s["prefix"][Ps:] for s in chunk],
                                                    [s["pv"][s.get("static_visuals", 0):] for s in chunk],
                                                    [s["positions"][:, Ps:P] if self.native else None for s in chunk],
                                                    cache, Ps)
                    jobs = [(r, s, q, suffix) for r, s in enumerate(chunk) for q, suffix in zip(s["qs"], s["sufs"])]
                    sb = int(self.suffix_batch_max or 0) if not grad else 0
                    shared = self.shared_prefix_attn and not grad
                    if not grad and (shared or self.sort_suffixes):
                        jobs.sort(key=lambda j: len(j[3]))   # V1 (2026-11-03): equal lengths in a sub-batch, no padding
                    if not shared and (not sb or sb >= len(jobs)) and (grad or not self.sort_suffixes):
                        repeats = torch.tensor([len(s["sufs"]) for s in chunk], device=device)
                        self._repeat_cache(cache, repeats)
                        self._letter_suffix_pass(jobs, cache, P, logits, weight, device)
                    else:                  # A4 (2026-11-03): bounded copies of the prefix-cache rows per sub-batch
                        ctx = self._shared_prefix_context(cache) if shared else None
                        for start in range(0, len(jobs), sb or len(jobs)):
                            sub = jobs[start:start + (sb or len(jobs))]
                            rows = torch.tensor([r for r, _, _, _ in sub], device=device)
                            if shared:     # V2: only the linear-attention states are per-row copies
                                sub_cache = self._suffix_cache(cache, rows)
                                self._letter_suffix_pass(sub, sub_cache, P, logits, weight, device, ctx, rows)
                            else:
                                sub_cache = self._map_letter_caches([cache], lambda ts: ts[0].index_select(0, rows))
                                self._letter_suffix_pass(sub, sub_cache, P, logits, weight, device)
                            del sub_cache
                    for s in chunk:
                        visual = len(s["pv"]) * self.tokens_per_frame
                        suffix_n = sum(map(len, s["sufs"]))
                        stats["paths"] += len(s["sufs"])
                        stats["prefix_tokens"] += P
                        stats["suffix_tokens"] += suffix_n
                        stats["vis_tokens"] += visual
                        stats["text_tokens"] += P - visual + suffix_n
                        stats["max_path"] = max(stats["max_path"], P + max(map(len, s["sufs"])))
        return logits, stats

    def _letter_suffix_pass(self, jobs, cache, P, logits, weight, device, ctx=None, rows=None):
        """One forward of question suffixes (right-padded; their resulting cache is discarded) on `cache`, whose rows
        are the jobs' prefix states in job order. jobs: [(chunk row, spec, (instruction, idx), suffix ids)].
        ctx / rows (V2, policy.shared_prefix_attn): `cache` holds only the rows' linear-attention states; the full-attention
        layers read the chunk's prefix KV in place (ctx, one copy per prefix; rows = each job's prefix row)."""
        paths = [suffix for _, _, _, suffix in jobs]
        embeds, mask, lengths = self._embed_letter_batch(paths, [[] for _ in jobs], device)
        if ctx is None:
            attn = torch.cat([torch.ones(len(jobs), P, dtype=mask.dtype, device=device), mask], dim=1)
        else:      # full attention handles causality itself; linear attention zeroes the suffix padding (if any)
            attn = {"full_attention": None, "linear_attention": None if bool(mask.all()) else mask}
        if self.native:
            pos = torch.zeros(len(jobs), embeds.shape[1], dtype=torch.long)
            for i, (_, s, _, suffix) in enumerate(jobs):
                pos[i, :len(suffix)] = s["next_pos"] + torch.arange(len(suffix))    # text: t = h = w
            pos = pos.to(device)[None].repeat(3, 1, 1)
        else:
            pos = (P + torch.arange(embeds.shape[1], device=device))[None].expand(len(jobs), -1)
        with _shared_prefix_attention(self.backbone, ctx, rows):
            h = self.backbone(inputs_embeds=embeds, attention_mask=attn, position_ids=pos,
                              past_key_values=cache, use_cache=True).last_hidden_state
        leaves = h[torch.arange(len(jobs), device=device), lengths - 1]
        self._letter_logits_into(logits, leaves, [(s["b"], q[1]) for _, s, q, _ in jobs], weight, device)

    @staticmethod
    def _shared_prefix_context(cache):
        """V2: the chunk's full-attention prefix KV as stored ([n, Hkv, P, D]) and, on GPU, in the flash layout
        ([n, P, Hkv, D], transposed once per chunk and shared by all its suffix sub-batches)."""
        kv, flash = {}, {}
        for i, layer in enumerate(cache.layers):
            k = getattr(layer, "keys", None)
            if torch.is_tensor(k) and k.numel():
                kv[i] = (k, layer.values)
                if k.is_cuda:
                    flash[i] = (k.transpose(1, 2).contiguous(), layer.values.transpose(1, 2).contiguous())
        return {"kv": kv, "flash_kv": flash}

    @staticmethod
    def _suffix_cache(cache, rows):
        """V2: the suffix rows' cache. Linear-attention conv / recurrent states are per-row copies (small); full-attention
        layers start empty, so update() stores only the suffix and the shared-prefix attention reads the prefix in place."""
        import copy
        result = copy.copy(cache)
        result.layers = []
        for template in cache.layers:
            layer = copy.copy(template)
            for name, value in vars(template).items():
                if isinstance(value, (dict, list)):
                    setattr(layer, name, value.copy())
            k = getattr(template, "keys", None)
            if torch.is_tensor(k) and k.numel():
                layer.is_initialized, layer.keys, layer.values = False, None, None
            for name in ("conv_states", "recurrent_states"):
                states = getattr(template, name, None)
                if isinstance(states, (dict, list)):
                    for j in (states.keys() if isinstance(states, dict) else range(len(states))):
                        if torch.is_tensor(states[j]):
                            getattr(layer, name)[j] = states[j].index_select(0, rows)
            result.layers.append(layer)
        return result

    def _letter_one_pass(self, s, logits, stats, weight, device):
        """A2 (2026-11-03, policy.one_pass_single): a no-grad single-question decision with a static snapshot. The rest
        of its prefix and its question suffix run as ONE forward on a copy of the snapshot (the cache after it is
        discarded): the same tokens, positions and attention as the prefix-rest forward + suffix forward, one pass."""
        Ps, P, suffix = s["static_length"], len(s["prefix"]), s["sufs"][0]
        cache = self._map_letter_caches([s["snapshot"]], lambda ts: torch.cat(ts, dim=0))
        tokens = s["prefix"][Ps:] + suffix
        embeds, _, _ = self._embed_letter_batch([tokens], [s["pv"][s.get("static_visuals", 0):]], device)
        if self.native:
            tail = (s["next_pos"] + torch.arange(len(suffix)))[None].expand(3, -1)
            pos = torch.cat([s["positions"][:, Ps:P], tail], dim=1)[:, None, :].to(device)          # [3, 1, L]
        else:
            pos = torch.arange(Ps, P + len(suffix), device=device)[None]
        h = self.backbone(inputs_embeds=embeds, attention_mask=torch.ones(1, P + len(suffix), dtype=torch.long,
                                                                          device=device),
                          position_ids=pos, past_key_values=cache, use_cache=True).last_hidden_state
        self._letter_logits_into(logits, h[:, -1], [(s["b"], s["qs"][0][1])], weight, device)
        visual = len(s["pv"]) * self.tokens_per_frame
        stats["paths"] += 1
        stats["prefix_tokens"] += P
        stats["suffix_tokens"] += len(suffix)
        stats["vis_tokens"] += visual
        stats["text_tokens"] += P - visual + len(suffix)
        stats["max_path"] = max(stats["max_path"], P + len(suffix))

    # ---------------------------------------------------------------- static-prefix KV cache (no grad; 2026-10-06)
    def clear_static_cache(self):
        """Drop every cached static head when training starts or weights are loaded (and any merged LoRA weights)."""
        self._static_kv = collections.OrderedDict()
        self._merge_version = getattr(self, "_merge_version", 0) + 1
        if "backbone" in self._modules:
            from .lora import release_merged
            release_merged(self.backbone)

    def _static_then_dynamic(self, d, ns, prefix, pv, pos, P, device, dt, DynamicCache):
        """policy.static_prefix_cache: the first d.static_segments prefix segments (task, question, options, overview
        frames; EpisodeV3._prefix) are computed once per (ids, frames) and kept as a snapshot that is never written to;
        every decision copies the snapshot and computes only the rest of its prefix. Positions (incl. M-RoPE) are the
        slices of the full path's positions, so the result equals the uncached forward up to kernel numerics
        (tests/test_static_prefix_cache.py)."""
        import copy
        pvs = []
        sids = self._segments_to_ids(d.prefix[:ns], pvs)
        Ps = len(sids)
        assert prefix[:Ps] == sids and Ps < P, "static head must be a strict token prefix"
        key = (tuple(sids), tuple((r.content_id, r.pts) for r in pvs))
        if not hasattr(self, "_static_kv"):
            self.clear_static_cache()
        snap = self._static_kv.get(key)
        if snap is None:
            snap = DynamicCache(config=self.backbone.config)
            emb = self._embed(sids, pvs, device)[None].to(dt) if pvs else self._embed(sids, [], device)[None]
            kw = {"position_ids": pos[:, None, :Ps].to(device)} if self.native else {}
            self.backbone(inputs_embeds=emb, attention_mask=torch.ones(1, Ps, dtype=torch.long, device=device),
                          past_key_values=snap, use_cache=True, **kw)
            self._static_kv[key] = snap
            while len(self._static_kv) > self.static_cache_max:
                self._static_kv.popitem(last=False)
            self.static_cache_stats["miss"] += 1
        else:
            self._static_kv.move_to_end(key)
            self.static_cache_stats["hit"] += 1
        cache = copy.deepcopy(snap)
        rest, rest_v = prefix[Ps:], pv[len(pvs):]
        emb = self._embed(rest, rest_v, device)[None].to(dt) if rest_v else self._embed(rest, [], device)[None]
        kw = ({"position_ids": pos[:, None, Ps:P].to(device)} if self.native else
              {"position_ids": (Ps + torch.arange(P - Ps, device=device))[None]})
        self.backbone(inputs_embeds=emb, attention_mask=torch.ones(1, P, dtype=torch.long, device=device),
                      past_key_values=cache, use_cache=True, **kw)
        return cache

    def _project(self, refs, device):
        if self.visual_mode == "zero":   # same slots, positions and mask; no visual content, no projector grad
            return torch.zeros(len(refs) * self.tokens_per_frame, self.projector.fc2.out_features, device=device)
        src = self.token_store if self.token_store is not None else self.feature_store
        feats = src.get([(r.content_id, r.pts) for r in refs]).to(device)   # [N, T, D]
        if feats.dim() == 2:
            feats = feats[:, None, :]
        assert feats.shape[1] == self.tokens_per_frame, (feats.shape, self.tokens_per_frame)
        if self.native:   # already LM-space tokens (frozen Qwen3.5 merger output)
            out = feats.reshape(-1, feats.shape[-1])
            assert out.shape == (len(refs) * self.tokens_per_frame, self.backbone.config.hidden_size), out.shape
            return out
        return self.projector(feats.reshape(-1, feats.shape[-1]))


def _update_conv_state_oop(self, conv_states, state_idx=0, conv_kernel_size=None, **kwargs):
    """transformers 5.17 LinearAttentionLayer.update_conv_state without its in-place copy_ (kept there for a static
    cudagraph address), which would overwrite a tensor that autograd saved for the backward."""
    assert not self.record_past
    if not self.is_conv_states_initialized[state_idx]:
        self.lazy_initialization(conv_states=conv_states, state_idx=state_idx, conv_kernel_size=conv_kernel_size)
    if not self.has_previous_state[state_idx]:
        full = conv_states
        self.has_previous_state[state_idx] = True
        if full.shape[-1] < self.conv_kernel_size[state_idx]:
            full = F.pad(full, (self.conv_kernel_size[state_idx] - full.shape[-1], 0), value=0)
    else:
        full = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)
    self.conv_states[state_idx] = full[..., -self.conv_kernel_size[state_idx]:]
    return full


def _update_recurrent_state_oop(self, recurrent_states, state_idx=0, **kwargs):
    """transformers 5.17 LinearAttentionLayer.update_recurrent_state without the in-place copy_."""
    if not self.is_recurrent_states_initialized[state_idx]:
        self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
    self.recurrent_states[state_idx] = recurrent_states
    return recurrent_states


_OUT_OF_PLACE_CLASSES = {}


def out_of_place_linear_cache(cache):
    """Make a DynamicCache's linear-attention (Gated DeltaNet) layers update their states by assignment, so a forward
    with gradients can run through it; full-attention layers already concatenate out of place. The layers get a
    subclass that carries the two methods (methods bound on the instance would form a layer <-> method reference cycle,
    so the layer's states would outlive the cache until Python's cyclic GC runs)."""
    for layer in cache.layers:
        cls = type(layer)
        if hasattr(cls, "update_conv_state") and not getattr(cls, "_out_of_place", False):
            sub = _OUT_OF_PLACE_CLASSES.get(cls)
            if sub is None:
                sub = _OUT_OF_PLACE_CLASSES[cls] = type(f"OutOfPlace{cls.__name__}", (cls,), {
                    "update_conv_state": _update_conv_state_oop, "update_recurrent_state": _update_recurrent_state_oop,
                    "_out_of_place": True})
            layer.__class__ = sub
    return cache


def apply_policy_sdpa_backend(cfg):
    """policy.sdpa_backend=math (2026-09-30, design B): process-wide math SDPA, set before any forward. The flash /
    mem-efficient SDPA backward is not deterministic (Qwen3.5-4B, 8-path decision: two identical backward passes
    differ by global grad rel-L2 0.046); with math they are bit-identical. The fla Gated DeltaNet kernels were
    deterministic in the same test."""
    b = cfg["policy"].get("sdpa_backend")
    if b in (None, "default"):
        return
    assert b == "math", b
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


# 2026-11-03 V2 (policy.shared_prefix_attn): question suffixes attend to their prefix KV in place instead of one copied
# KV per suffix row (the 96-question overview decision spent ~30% of its GPU time copying / concatenating prefix KV).
_SHARED_CTX = {}


def _shared_prefix_attention_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """Attention of suffix queries over [their prefix (one stored copy per prefix; _SHARED_CTX rows maps every suffix row
    to its prefix row); their own suffix, causally]. GPU: flash-attn over the prefix (kv-cache API, cache_batch_idx) and
    over the suffix (causal), merged by log-sum-exp; elsewhere the exact causal attention over the concatenation.
    query [B, Hq, S, D]; key / value: the suffix's own [B, Hkv, S, D]. Returns ([B, S, Hq, D], None). Padded suffix
    positions come after the valid ones, so causal attention never lets a valid query see them."""
    ctx, rows = _SHARED_CTX["ctx"], _SHARED_CTX["rows"]
    B, Hq, S, _ = query.shape
    if query.is_cuda and module.layer_idx in ctx["flash_kv"]:
        from flash_attn import flash_attn_func, flash_attn_with_kvcache
        kc, vc = ctx["flash_kv"][module.layer_idx]
        q = query.transpose(1, 2).contiguous()
        o1, l1 = flash_attn_with_kvcache(q, kc, vc, cache_seqlens=torch.full((B,), kc.shape[1], dtype=torch.int32,
                                                                              device=q.device),
                                         cache_batch_idx=rows.to(torch.int32), softmax_scale=scaling, causal=False,
                                         return_softmax_lse=True)
        o2, l2, _ = flash_attn_func(q, key.transpose(1, 2).contiguous(), value.transpose(1, 2).contiguous(),
                                    softmax_scale=scaling, causal=True, return_attn_probs=True)
        m = torch.maximum(l1, l2)
        w1, w2 = (l1 - m).exp().transpose(1, 2)[..., None], (l2 - m).exp().transpose(1, 2)[..., None]
        return ((o1.float() * w1 + o2.float() * w2) / (w1 + w2)).to(query.dtype), None
    pk, pv = ctx["kv"][module.layer_idx]
    P = pk.shape[2]
    k = torch.cat([pk.index_select(0, rows), key], dim=2)
    v = torch.cat([pv.index_select(0, rows), value], dim=2)
    g = Hq // k.shape[1]
    k, v = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
    scores = (query.float() @ k.float().transpose(-1, -2)) * scaling
    allowed = torch.ones(S, P + S, dtype=torch.bool, device=query.device).tril(diagonal=P)
    out = scores.masked_fill(~allowed, float("-inf")).softmax(-1) @ v.float()
    return out.to(query.dtype).transpose(1, 2).contiguous(), None


@contextlib.contextmanager
def _shared_prefix_attention(backbone, ctx, rows):
    """Route the backbone's attention through _shared_prefix_attention_forward for one forward (no-op for ctx None)."""
    if ctx is None:
        yield
        return
    from transformers.modeling_utils import AttentionInterface
    if "jev_shared_prefix" not in AttentionInterface._global_mapping:
        AttentionInterface.register("jev_shared_prefix", _shared_prefix_attention_forward)
    cfg = backbone.config
    old = cfg._attn_implementation
    _SHARED_CTX.update(ctx=ctx, rows=rows)
    cfg._attn_implementation = "jev_shared_prefix"
    try:
        yield
    finally:
        cfg._attn_implementation = old
        _SHARED_CTX.clear()


def payload_groups(d):
    """Grouped decision (2026-10-06, joint per-frame verifier): every payload carries "group" -> list of group ids in
    candidate order (each group is normalized on its own); else None. d: DecisionInput or its json."""
    pls = d.payloads if hasattr(d, "payloads") else d.get("payloads", [])
    if pls and all(isinstance(pl, dict) and "group" in pl for pl in pls):
        return [pl["group"] for pl in pls]
    return None


def grouped_log_softmax(z, groups):
    """z: [K] logits (-inf = illegal); log-softmax inside each group; a group without a legal candidate stays -inf."""
    out = torch.full_like(z.float(), float("-inf"))
    for g in dict.fromkeys(groups):
        idx = torch.tensor([i for i, x in enumerate(groups) if x == g], device=z.device)
        out[idx] = masked_log_softmax(z[idx].float()[None])[0]
    return out


def masked_log_softmax(logits):
    """Rows whose entries are all -inf stay -inf (no legal action)."""
    safe = torch.where(torch.isinf(logits).all(-1, keepdim=True), torch.zeros_like(logits), logits)
    out = safe.float().log_softmax(-1)
    return torch.where(torch.isinf(logits), torch.full_like(out, float("-inf")), out)


# -------------------------------------------------------------------- loading
def load_backbone_and_tokenizer(cfg_models, init="nanojev", attn_impl="sdpa", qwen35_model="qwen35_4b"):
    """init='nanojev': NanoJev unified-games-v1 backbone. init='qwen3_fresh': Qwen3-0.6B weights.
    init='qwen35_4b': Qwen3.5-4B language model (own tokenizer)."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    if init != "qwen35_4b":          # the NanoJev files are needed only by the 0.6B variants
        nj = cfg_models["nanojev"]["path"]
        tok = AutoTokenizer.from_pretrained(f"{nj}/tokenizer")
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
    if init == "nanojev":
        conf = AutoConfig.from_pretrained(f"{nj}/backbone_config")
        conf.use_cache = False
        body = AutoModel.from_config(conf, attn_implementation=attn_impl).float()
    elif init == "qwen35_4b":       # 2026-09-30: Qwen3.5-4B (hybrid Gated DeltaNet / attention); the language model only,
        # frozen base weights in bf16 (LoRA-trained); frames enter as projected SigLIP tokens, or (frame_tokens=
        # qwen_native) as tokens precomputed with its own vision tower (native_token_store), so the tower is not kept
        path = cfg_models[qwen35_model]["path"]         # policy.qwen35_model: qwen35_4b (default) | qwen35_9b
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        full = AutoModel.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=attn_impl)
        body = full.language_model
        del full
        body.config.use_cache = False
    elif init == "qwen3_fresh":
        body = AutoModel.from_pretrained(cfg_models["qwen3_0p6b"]["path"], dtype=torch.float32,
                                         attn_implementation=attn_impl)
        body.config.use_cache = False
    else:
        raise ValueError(init)
    return body, tok


def configure_finetune(policy, pcfg):
    """'lora' (default): LoRA-wrap the backbone (B=0: function unchanged) and freeze its base weights.
    'full': no LoRA, every backbone weight trainable. Projector and head are trainable in both."""
    mode = pcfg.get("finetune", "lora")
    if mode == "lora":
        lc = pcfg["lora"]
        policy.lora_modules = apply_lora(policy.backbone, lc["targets"], lc["r"], lc["alpha"], lc["dropout"])
    elif mode == "full":
        policy.lora_modules = []
        policy.backbone.requires_grad_(True)
    else:
        raise ValueError(f"policy.finetune must be 'lora' or 'full', got {mode!r}")
    policy.finetune = mode
    return policy


def build_policy(cfg, feature_store=None, init=None, device="cpu", with_lora=True):
    """Base weights (+ NanoJev head) -> LoRA wrap (B=0: function unchanged) or full finetune
    (cfg policy.finetune) -> projector/head."""
    if cfg["policy"].get("compile_norms"):              # 2026-11-05 F1 (src/policy/fused.py)
        from .fused import compile_qwen35_norms
        compile_qwen35_norms()
    from safetensors.torch import load_file
    init = init or cfg["policy"]["init"]
    # policy.attn_implementation (2026-09-30): flash_attention_2 (flash-attn package, env jev_fa) with
    # FLASH_ATTENTION_DETERMINISTIC=1 gives a deterministic backward; the default sdpa's flash backward is not
    body, tok = load_backbone_and_tokenizer(cfg["models"], init, attn_impl=cfg["policy"].get("attn_implementation") or "sdpa",
                                         qwen35_model=cfg["policy"].get("qwen35_model", "qwen35_4b"))
    head_sd = None
    if init == "nanojev":
        sd = load_file(f"{cfg['models']['nanojev']['path']}/best.safetensors", device="cpu")
        body.load_state_dict({k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")},
                             strict=True)
        head_sd = {k: v for k, v in sd.items() if not k.startswith("backbone.")}
        assert set(k.split(".")[0] for k in head_sd) <= set(HEAD_KEYS), sorted(head_sd)[:5]
    # pooled (grid^2 tokens) | global1 (1 token) | qwen_native ((native_side/32)^2 Qwen3.5 image tokens, init=qwen35_4b)
    frame_tokens = cfg["policy"].get("frame_tokens", "pooled")
    if frame_tokens == "qwen_native":
        assert init == "qwen35_4b", "frame_tokens=qwen_native needs init=qwen35_4b"
        side = cfg["policy"].get("native_side", 256)
        assert side % 32 == 0, side          # patch 16 x spatial merge 2
        tpf = (side // 32) ** 2
    else:
        tpf = 1 if frame_tokens == "global1" else cfg["video"]["pool_grid"] ** 2
    policy = JevPolicy(body, tok, vis_dim=cfg["policy"]["vis_dim"], tokens_per_frame=tpf)
    if head_sd is not None:
        policy.head.load_state_dict(head_sd, strict=True)
    if with_lora:
        configure_finetune(policy, cfg["policy"])
    policy.feature_store = feature_store
    if frame_tokens == "global1":   # frame tokens from the official global embeddings; the prior keeps feature_store
        from ..rollout.build import global_token_store
        policy.token_store = global_token_store(cfg)[1]
    if frame_tokens == "qwen_native":
        from ..rollout.build import native_token_store
        policy.token_store = native_token_store(cfg)[1]
        policy.native = True
        policy.vision_start_id = tok.convert_tokens_to_ids("<|vision_start|>")
        policy.vision_end_id = tok.convert_tokens_to_ids("<|vision_end|>")
        assert (policy.vision_start_id, policy.vision_end_id) == (248053, 248054), (policy.vision_start_id, policy.vision_end_id)
        policy.projector.requires_grad_(False)     # unused
    policy.strip_vestigial = bool(cfg["policy"].get("strip_vestigial", False))
    policy.prefix_sharing = bool(cfg["policy"].get("prefix_sharing", False))
    policy.prefix_sharing_train = bool(cfg["policy"].get("prefix_sharing_train", False))
    policy.static_cache = bool(cfg["policy"].get("static_prefix_cache", False))
    policy.static_cache_max = int(cfg["policy"].get("static_prefix_cache_max", 32))
    policy.deduplicate_inference = bool(cfg["policy"].get("deduplicate_inference", True))
    policy.readout = cfg["policy"].get("readout", "head")
    policy.forward_max_paths = int(cfg["policy"].get("max_paths_per_forward", 64))
    policy.forward_max_tokens = int(cfg["policy"].get("max_tokens_per_forward", 98304))
    policy.one_pass_single = bool(cfg["policy"].get("one_pass_single", False))           # 2026-11-03 A2
    policy.suffix_batch_max = int(cfg["policy"].get("suffix_batch_max", 0) or 0)        # 2026-11-03 A4
    policy.merge_lora = bool(cfg["policy"].get("merge_lora_inference", False))           # 2026-11-03 A1
    from .lora import set_merge_per_role                                                  # 2026-11-05 E1
    set_merge_per_role(policy.merge_lora and bool(cfg["policy"].get("merge_lora_per_role", False)))
    policy.sort_suffixes = bool(cfg["policy"].get("sort_suffixes", False))               # 2026-11-03 V1
    policy.shared_prefix_attn = bool(cfg["policy"].get("shared_prefix_attn", False))     # 2026-11-03 V2
    if policy.readout == "letters":                 # OneJev: no DecisionHead, no projector (native tokens)
        assert not cfg["policy"].get("candidate_vis") and not cfg["policy"].get("siglip_prior"), "letters readout"
        policy.head.requires_grad_(False)
    policy.visual_mode = cfg["policy"].get("visual_mode", "normal")
    policy.candidate_vis = bool(cfg["policy"].get("candidate_vis", False))
    policy = policy.to(device)
    if cfg["policy"].get("siglip_prior"):
        policy.enable_prior(PriorScorer(cfg["models"]["siglip2"]["path"], device=device),
                            gate_init=cfg["policy"].get("prior_gate_init", 1.0))
    return policy


def share_frozen_backbone(policy, source):
    """Share the immutable base of two LoRA policies loaded from the same base model.

    LoRA tensors, heads, projectors and caches remain independent. Full fine-tunes
    cannot share their backbone, since those base parameters can change.
    """
    if policy.finetune != 'lora' or source.finetune != 'lora':
        return 0
    src = dict(source.backbone.named_parameters())
    shared_bytes = 0
    for name, parameter in list(policy.backbone.named_parameters()):
        other = src.get(name)
        if parameter.requires_grad or 'lora_' in name or other is None or other.requires_grad:
            continue
        if (parameter.shape, parameter.dtype, parameter.device) != (other.shape, other.dtype, other.device):
            raise ValueError(f'incompatible frozen backbone parameter: {name}')
        parent, _, leaf = name.rpartition('.')
        policy.backbone.get_submodule(parent)._parameters[leaf] = other
        shared_bytes += parameter.numel() * parameter.element_size()
    return shared_bytes


def frozen_copy(policy):
    """Independent reference adapters, sharing immutable base parameters and feature stores."""
    import copy
    store, policy.feature_store = policy.feature_store, None
    tstore, policy.token_store = policy.token_store, None
    scorer, policy.prior_scorer = policy.prior_scorer, None
    static = getattr(policy, '_static_kv', None)
    policy.clear_static_cache()  # references start cold; do not duplicate large inference snapshots
    try:
        memo = {id(p): p for p in policy.backbone.parameters() if not p.requires_grad}
        ref = copy.deepcopy(policy, memo)
    finally:
        policy.feature_store, policy.prior_scorer, policy.token_store = store, scorer, tstore
        if static is not None:
            policy._static_kv = static
    ref.feature_store, ref.prior_scorer, ref.token_store = store, scorer, tstore
    ref.eval().requires_grad_(False)
    return ref
