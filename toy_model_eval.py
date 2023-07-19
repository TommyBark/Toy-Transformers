# %%
import torch
from transformers import GPT2TokenizerFast
from toytransformer.transformer import TransformerSampler, LitTransformer
from toytransformer.utils import fix_compiled_model_ckpt
import pytorch_lightning as pl
from typing import List, Optional
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = ".\checkpoints\epoch=4-step=9774.ckpt"


tokenizer = GPT2TokenizerFast(
    vocab_file="./tiny_tokenizer/tiny_stories_10k_tokenizer-vocab.json",
    merges_file="./tiny_tokenizer/tiny_stories_10k_tokenizer-merges.txt",
)


@dataclass
class TransformerTrainingArgs:
    batch_size = 4096 * 2
    max_epochs = 3
    max_steps = 1000
    log_every = 10
    lr = 1e-3
    weight_decay = 1e-2
    log_dir: str = "/logs/"
    log_name: str = "toy_model"
    run_name: Optional[str] = None
    log_every_n_steps: int = 1


args = TransformerTrainingArgs()


fix_compiled_model_ckpt(checkpoint_path, device)
model = LitTransformer.load_from_checkpoint(
    checkpoint_path.replace(".ckpt", "_fixed.ckpt"),
    args=args,
    map_location=device,
    strict=True,
)
# model = LitTransformer.load(checkpoint_path, map_location=device, strict=False)
model.eval()

sampler = TransformerSampler(model, tokenizer)

prompt = "One upon a time, there was"

sampler.sample(prompt, max_tokens_generated=20)

# %%
