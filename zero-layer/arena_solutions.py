# %%

import os

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
import einops
from dataclasses import dataclass
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
import torch as t
from torch import Tensor
import torch.nn as nn
import numpy as np
import math
from tqdm.notebook import tqdm
from typing import Tuple, List, Optional, Dict
from jaxtyping import Float, Int
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from collections import defaultdict
from rich.table import Table
from rich import print as rprint
import datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pathlib import Path
import webbrowser

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part1_transformer_from_scratch").resolve()
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from plotly_utils import imshow

# import part1_transformer_from_scratch.solutions as solutions

device = t.device("cuda" if t.cuda.is_available() else "cpu")


@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


# %%


def rand_float_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randn(shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple):
        output = output[0]
    print("Output shape:", output.shape, "\n")


def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randint(100, 1000, shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple):
        output = output[0]
    print("Output shape:", output.shape, "\n")


def load_gpt2_test(cls, gpt2_layer, input):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    print("Input shape:", input.shape)
    output = layer(input)
    if isinstance(output, tuple):
        output = output[0]
    print("Output shape:", output.shape)
    try:
        reference_output = gpt2_layer(input)
    except:
        reference_output = gpt2_layer(input, input, input)
    print("Reference output shape:", reference_output.shape, "\n")
    comparison = t.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f"{comparison.sum()/comparison.numel():.2%} of the values are correct\n")


# %%


class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        residual_mean = residual.mean(dim=-1, keepdim=True)
        residual_std = (
            residual.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps
        ).sqrt()

        residual = (residual - residual_mean) / residual_std
        return residual * self.w + self.b


# %%


class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        return self.W_E[tokens]


# %%


class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        batch, seq_len = tokens.shape
        return einops.repeat(
            self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch=batch
        )


# %%


class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(-1e5, dtype=t.float32, device=device))

    def forward(
        self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # Calculate query, key and value vectors
        q = (
            einops.einsum(
                normalized_resid_pre,
                self.W_Q,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_Q
        )
        k = (
            einops.einsum(
                normalized_resid_pre,
                self.W_K,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_K
        )
        v = (
            einops.einsum(
                normalized_resid_pre,
                self.W_V,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_V
        )

        # Calculate attention scores, then scale and mask, and apply softmax to get probabilities
        attn_scores = einops.einsum(
            q,
            k,
            "batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K",
        )
        attn_scores_masked = self.apply_causal_mask(
            attn_scores / self.cfg.d_head**0.5
        )
        attn_pattern = attn_scores_masked.softmax(-1)

        # Take weighted sum of value vectors, according to attention probabilities
        z = einops.einsum(
            v,
            attn_pattern,
            "batch posn_K nheads d_head, batch nheads posn_Q posn_K -> batch posn_Q nheads d_head",
        )

        # Calculate output (by applying matrix W_O and summing over heads, then adding bias b_O)
        attn_out = (
            einops.einsum(
                z,
                self.W_O,
                "batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model",
            )
            + self.b_O
        )

        return attn_out

    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        # Define a mask that is True for all positions we want to set probabilities to zero for
        all_ones = t.ones(
            attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device
        )
        mask = t.triu(all_ones, diagonal=1).bool()
        # Apply the mask to attention scores, then return the masked scores
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


# %%


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        pre = (
            einops.einsum(
                normalized_resid_mid,
                self.W_in,
                "batch position d_model, d_model d_mlp -> batch position d_mlp",
            )
            + self.b_in
        )
        post = gelu_new(pre)
        mlp_out = (
            einops.einsum(
                post,
                self.W_out,
                "batch position d_mlp, d_mlp d_model -> batch position d_model",
            )
            + self.b_out
        )
        return mlp_out


# %%


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(
        self, resid_pre: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post


# %%


class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        return (
            einops.einsum(
                normalized_resid_final,
                self.W_U,
                "batch posn d_model, d_model d_vocab -> batch posn d_vocab",
            )
            + self.b_U
        )
        # Or, could just do `normalized_resid_final @ self.W_U + self.b_U`


# %%


class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        residual = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        logits = self.unembed(self.ln_final(residual))
        return logits


# %%


def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = (
        log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    )

    return log_probs_for_tokens


# %% 3️⃣ TRAINING A TRANSFORMER
@dataclass
class TransformerTrainingArgs:
    batch_size = 8
    max_epochs = 1
    max_steps = 1000
    log_every = 10
    lr = 1e-3
    weight_decay = 1e-2
    log_dir: str = os.getcwd() + "/logs"
    log_name: str = "day1-transformer"
    run_name: Optional[str] = None
    log_every_n_steps: int = 1


# %%


class LitTransformer(pl.LightningModule):
    def __init__(
        self,
        args: TransformerTrainingArgs,
        model: DemoTransformer,
        data_loader: DataLoader,
    ):
        super().__init__()
        self.model = model
        self.cfg = model.cfg
        self.args = args
        self.data_loader = data_loader

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        logits = self.model(tokens)
        return logits

    def training_step(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Float[Tensor, ""]:
        """
        Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        """
        tokens = batch["tokens"].to(device)
        logits = self.model(tokens)
        loss = -get_log_probs(logits, tokens).mean()
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        """
        optimizer = t.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        return optimizer

    def train_dataloader(self):
        return self.data_loader


# %% 4️⃣ SAMPLING FROM A TRANSFORMER


# %%


class TransformerSampler:
    def __init__(self, model: DemoTransformer, tokenizer: GPT2TokenizerFast):
        self.model = model
        self.cfg = model.cfg
        self.tokenizer = tokenizer

    @t.inference_mode()
    def sample(self, prompt: str, max_tokens_generated=100, verbose=False, **kwargs):
        """
        Returns a string of autoregressively generated text, starting from the prompt.

        Sampling terminates at max_tokens_generated, or when the model generates an
        end-of-sequence token.

        kwargs are passed to sample_next_token, to give detailed instructions on how
        new tokens are chosen.
        """
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)[0]

        for i in range(max_tokens_generated):
            # Get new logits (make sure we don't pass in more tokens than the model's context length)
            logits = self.model(input_ids[None, -self.cfg.n_ctx :])
            # We only take logits for the last token, because this is what we're sampling
            logits = logits[0, -1]
            # Get next token (as a tensor of size (1, 1) so we can concat it to input_ids)
            next_token = t.tensor(
                [TransformerSampler.sample_next_token(input_ids, logits, **kwargs)],
                device=device,
            )
            # Create new input ids string, with shape (1, old_seq_len + 1)
            input_ids = t.cat([input_ids, next_token], dim=-1)
            # Print out results, if required
            if verbose:
                print(self.tokenizer.decode(input_ids), end="\r")
            # If our new token was the end-of-text token, stop
            if next_token == getattr(self.tokenizer, "eos_token_id", None):
                break

        return self.tokenizer.decode(input_ids)

    @t.inference_mode()
    def beam_search(
        self,
        prompt: str,
        num_return_sequences: int,
        num_beams: int,
        max_new_tokens: int,
        no_repeat_ngram_size: int = 0,
        verbose=False,
    ) -> List[Tuple[float, t.Tensor]]:
        """
        Returns a string of autoregressively generated text, starting from the prompt.

        Sampling terminates at max_tokens_generated, or when the model generates an
        end-of-sequence token.

        kwargs are passed to sample_next_token, to give detailed instructions on how
        new tokens are chosen.
        """
        pass

    @staticmethod
    def sample_next_token(
        input_ids: Int[Tensor, "seq_len"],
        logits: Float[Tensor, "seq_len d_vocab"],
        temperature=1.0,
        top_k=0,
        top_p=0.0,
        frequency_penalty=0.0,
        seed=None,
    ):
        assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
        assert temperature >= 0, "Temperature should be non-negative"
        assert 0 <= top_p <= 1.0, "Top-p must be a probability"
        assert 0 <= top_k, "Top-k must be non-negative"
        assert not (
            top_p != 0 and top_k != 0
        ), "At most one of top-p and top-k supported"

        # Set random seeds for reproducibility
        if seed is not None:
            t.manual_seed(seed)
            np.random.seed(seed)

        # Apply all the specialized sampling methods
        if temperature == 0:
            return TransformerSampler.greedy_search(logits)
        elif temperature != 1.0:
            logits = TransformerSampler.apply_temperature(logits, temperature)
        if frequency_penalty != 0.0:
            logits = TransformerSampler.apply_frequency_penalty(
                input_ids, logits, frequency_penalty
            )
        if top_k > 0:
            return TransformerSampler.sample_top_k(logits, top_k)
        if top_p > 0.0:
            return TransformerSampler.sample_top_p(logits, top_p)
        return TransformerSampler.sample_basic(logits)

    @staticmethod
    def greedy_search(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Returns the most likely token (as an int).
        """
        out = logits.argmax().item()
        return out

    @staticmethod
    def apply_temperature(
        logits: Float[Tensor, "d_vocab"], temperature: float
    ) -> Float[Tensor, "d_vocab"]:
        """
        Applies temperature scaling to the logits.
        """
        return logits / temperature

    @staticmethod
    def apply_frequency_penalty(
        input_ids: Int[Tensor, "seq_len"],
        logits: Float[Tensor, "d_vocab"],
        freq_penalty: float,
    ) -> Float[Tensor, "d_vocab"]:
        """
        Applies a frequency penalty to the logits.
        """
        d_vocab = logits.size(0)
        id_freqs = t.bincount(input_ids, minlength=d_vocab)
        return logits - freq_penalty * id_freqs

    @staticmethod
    def sample_basic(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Samples from the distribution defined by the logits.
        """
        sampled_token = t.distributions.categorical.Categorical(logits=logits).sample()
        return sampled_token.item()

    @staticmethod
    def sample_top_k(logits: Float[Tensor, "d_vocab"], k: int) -> int:
        """
        Samples from the top k most likely tokens.
        """
        top_k_logits, top_k_token_ids = logits.topk(k)
        # Get sampled token (which is an index corresponding to the list of top-k tokens)
        sampled_token_idx = t.distributions.categorical.Categorical(
            logits=top_k_logits
        ).sample()
        # Get the actual token id, as an int
        return top_k_token_ids[sampled_token_idx].item()

    @staticmethod
    def sample_top_p(
        logits: Float[Tensor, "d_vocab"], top_p: float, min_tokens_to_keep: int = 1
    ) -> int:
        """
        Samples from the most likely tokens which make up at least p cumulative probability.
        """
        # Sort logits, and get cumulative probabilities
        logits_sorted, indices = logits.sort(descending=True, stable=True)
        cumul_probs = logits_sorted.softmax(-1).cumsum(-1)
        # Choose which tokens to keep, in the set we sample from
        n_keep = t.searchsorted(cumul_probs, top_p, side="left").item() + 1
        n_keep = max(n_keep, min_tokens_to_keep)
        keep_idx = indices[:n_keep]
        keep_logits = logits[keep_idx]
        # Perform the sampling
        sample = t.distributions.categorical.Categorical(logits=keep_logits).sample()
        return keep_idx[sample].item()


# %%
@dataclass
class Beams:
    """Class to store beams during beam search."""

    model: DemoTransformer
    tokenizer: GPT2TokenizerFast
    logprob_sums: Float[Tensor, "batch"]
    tokens: Int[Tensor, "batch seq"]

    def new_beams(self, logprob_sums, tokens) -> "Beams":
        """Creates a new Beams object with the same model and tokenizer."""
        return Beams(self.model, self.tokenizer, logprob_sums, tokens)

    def __getitem__(self, idx) -> "Beams":
        """Allows you to take a slice of the beams object along the batch dimension."""
        return self.new_beams(self.logprob_sums[idx], self.tokens[idx])

    @property
    def logprobs_and_completions(self) -> List[Tuple[float, str]]:
        """Returns self as a list of logprob sums and completions (useful for getting final output)."""
        return [
            (logprob_sum.item(), self.tokenizer.decode(tokens))
            for (logprob_sum, tokens) in zip(self.logprob_sums, self.tokens)
        ]

    def generate(
        self, toks_per_beam: int, no_repeat_ngram_size: Optional[int] = None
    ) -> "Beams":
        """
        Starting from the current set of beams (which has length `num_beams`), returns a new
        set of `num_beams * toks_per_beam`, containing the best `toks_per_beam` continuations for each
        of the original beams.

        Optional argument `no_repeat_ngram_size` means your model won't generate any sequences with
        a repeating n-gram of this length.
        """
        # Get the output logprobs for the next token (for every sequence in current beams)
        logprobs: Tensor = self.model(self.tokens)[:, -1, :].log_softmax(-1)

        # Get the top `toks_per_beam` tokens for each sequence
        # topk_logprobs, topk_tokenIDs = logprobs.topk(k=toks_per_beam)
        topk_logprobs, topk_tokenIDs = self.get_topk_non_repeating(
            logprobs, no_repeat_ngram_size, k=toks_per_beam
        )

        # Get all of the new possible beams, via einops operations
        #   Here, we're effectively flattening out the batch dimension and k dimension, to give us tensors
        #   with every possible combination of (original sequence, new token) pairs.)
        new_logprob_sums = sum(
            [
                einops.repeat(self.logprob_sums, "batch -> (batch k)", k=toks_per_beam),
                einops.rearrange(topk_logprobs, "batch k -> (batch k)"),
            ]
        )
        new_tokens = t.concat(
            [
                einops.repeat(
                    self.tokens, "batch seq -> (batch k) seq", k=toks_per_beam
                ),
                einops.rearrange(topk_tokenIDs, "batch k -> (batch k) 1"),
            ],
            dim=-1,
        )
        return self.new_beams(new_logprob_sums, new_tokens)

    def filter(self, num_beams: int) -> Tuple["Beams", "Beams"]:
        """
        Returns:
                best_beams: Beams
                        filtered version of self, containing all best `num_beams` which are also not terminated.

                early_terminations: Beams
                        filtered version of self, containing all best `num_beams` which are also terminated.
                        i.e. the sum of lengths of these two should equal `num_beams`.
        """
        # Get the indices of top `num_beams` beams
        top_beam_indices = self.logprob_sums.topk(k=num_beams, dim=0).indices.tolist()
        # Get the indices of terminated sequences
        new_tokens = self.tokens[:, -1]
        terminated_indices = t.nonzero(new_tokens == self.tokenizer.eos_token_id)

        # Get the indices of the `num_beams` best sequences (some terminated, some not terminated)
        best_continuing = [i for i in top_beam_indices if i not in terminated_indices]
        best_terminated = [i for i in top_beam_indices if i in terminated_indices]

        # Return the beam objects from these indices
        return self[best_continuing], self[best_terminated]

    def print(self, title="Best completions", max_print_chars=80) -> None:
        """
        Prints out a set of sequences with their corresponding logitsums.
        """
        if len(self.tokens) == 0:
            return
        table = Table("logitsum", "completion", title=title)
        for logprob_sum, tokens in zip(self.logprob_sums, self.tokens):
            text = self.tokenizer.decode(tokens)
            if len(repr(text)) > max_print_chars:
                text = (
                    text[: int(0.3 * max_print_chars)]
                    + " ... "
                    + text[-int(0.7 * max_print_chars) :]
                )
            table.add_row(f"{logprob_sum:>8.3f}", repr(text))
        rprint(table)

    def get_topk_non_repeating(
        self,
        logprobs: Float[Tensor, "batch d_vocab"],
        no_repeat_ngram_size: Optional[int],
        k: int,
    ) -> Tuple[Float[Tensor, "k"], Int[Tensor, "k"]]:
        """
        logprobs:
                tensor of the log-probs for the next token
        no_repeat_ngram_size:
                size of ngram to avoid repeating
        k:
                number of top logits to return, for each beam in our collection

        Returns:
                equivalent to the output of `logprobs.topk(dim=-1)`, but makes sure
                that no returned tokens would produce an ngram of size  `no_repeat_ngram_size`
                which has already appeared in `self.tokens`.
        """
        batch, seq_len = self.tokens.shape
        neg_inf = t.tensor(-1.0e4).to(device)

        # If completion isn't long enough for a repetition, or we have no restructions, just return topk
        if (no_repeat_ngram_size is not None) and (seq_len > no_repeat_ngram_size - 1):
            # Otherwise, we need to check for ngram repetitions
            # First, get the most recent `no_repeat_ngram_size-1` tokens
            last_ngram_prefix = self.tokens[:, seq_len - (no_repeat_ngram_size - 1) :]
            # Next, find all the tokens we're not allowed to generate (by going iterating through past ngrams and seeing if those ngram prefixes match the last one)
            for i in range(seq_len - (no_repeat_ngram_size - 1)):
                ngrams = self.tokens[:, i : i + no_repeat_ngram_size]  # (batch, ngram)
                ngrams_are_repeated = (ngrams[:, :-1] == last_ngram_prefix).all(
                    -1
                )  # (batch,)
                ngram_end_tokens = ngrams[:, [-1]]  # (batch, 1)
                # Fill logprobs with neginf wherever the ngrams are repeated
                logprobs[range(batch), ngram_end_tokens] = t.where(
                    ngrams_are_repeated,
                    neg_inf,
                    logprobs[range(batch), ngram_end_tokens],
                )

        # Finally, get our actual tokens
        return logprobs.topk(k=k, dim=-1)


@t.inference_mode()
def beam_search(
    self: TransformerSampler,
    prompt: str,
    num_return_sequences: int,
    num_beams: int,
    max_new_tokens: int,
    no_repeat_ngram_size: Optional[int] = None,
    verbose=False,
) -> List[Tuple[float, Tensor]]:
    """
    Implements a beam search, by repeatedly performing the `generate` and `filter` steps (starting
    from the initial prompt) until either of the two stopping criteria are met:

            (1) we've generated `max_new_tokens` tokens, or
            (2) we've generated `num_returns_sequences` terminating sequences.

    To modularize this function, most of the actual complexity is in the Beams class,
    in the `generate` and `filter` methods.
    """

    assert num_return_sequences <= num_beams
    self.model.eval()

    tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(device)

    # List for final beams to return (and early terminations)
    final_logprobs_and_completions: List[Tuple[float, str]] = []
    # Keep track of all best beams after each step
    best_beams = Beams(self.model, self.tokenizer, t.tensor([0.0]).to(device), tokens)

    for n in tqdm(range(max_new_tokens)):
        # Generation step
        best_beams = best_beams.generate(
            toks_per_beam=num_beams, no_repeat_ngram_size=no_repeat_ngram_size
        )

        # Filtering step
        best_beams, best_beams_terminated = best_beams.filter(num_beams=num_beams)
        final_logprobs_and_completions.extend(
            best_beams_terminated.logprobs_and_completions
        )

        # Print output
        if verbose:
            best_beams.print()

        # Check stopping condition
        if len(final_logprobs_and_completions) >= num_return_sequences:
            return final_logprobs_and_completions[:num_return_sequences]

    final_logprobs_and_completions.extend(best_beams.logprobs_and_completions)
    final_logprobs_and_completions = final_logprobs_and_completions[
        :num_return_sequences
    ]
    return final_logprobs_and_completions
