# %%
import os
import gc
from jaxtyping import Float, Int
from typing import List, Optional, Dict
from torch import Tensor
import torch.nn as nn
import torch
import datasets
from toytransformer.transformer import (
    TransformerBlock_noMLP,
    Embed,
    PosEmbed,
    Unembed,
    LayerNorm,
    get_log_probs,
)
from transformer_lens.utils import tokenize_and_concatenate
from torch.utils.data import DataLoader
from dataclasses import dataclass
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import pytorch_lightning as pl
from einops._torch_specific import allow_ops_in_compiled_graph

allow_ops_in_compiled_graph()


TOKENIZED_DATASET_PATH = "tokenized_dataset.hf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
n_cpus: int = os.cpu_count()
checkpoint_path = ".\checkpoints\epoch=4-step=9774.ckpt"
if n_gpus > 1:
    additional_training_kwargs = {
        "accelerator": "gpu",
        "devices": n_gpus,
        "strategy": "ddp_find_unused_parameters_true",
    }
else:
    additional_training_kwargs = {}

if n_cpus > 16:
    num_workers = n_cpus - 2
else:
    num_workers = 1

if device == torch.device("cuda"):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(False)
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True
    torch._dynamo.reset()
    torch._dynamo.config.verbose = True
    torch.set_float32_matmul_precision("medium")
    precision = 16
    # torch.cuda.empty_cache()
    gc.collect()
else:
    precision = "32-true"


@dataclass
class Config:
    d_model: int = 64
    debug: bool = False
    layer_norm_eps: float = 1e-5
    d_vocab: int = 10000
    init_range: float = 0.02
    n_ctx: int = 10
    d_head: int = 64
    n_heads: int = 6
    n_layers: int = 1


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
cfg = Config()


# %%
class ToyTransformer(pl.LightningModule):
    def __init__(
        self,
        cfg: Config,
        args: TransformerTrainingArgs,
    ):
        super().__init__()
        self.args = args
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList(
            [TransformerBlock_noMLP(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)
        self.save_hyperparameters()

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        x = self.embed(tokens) + self.pos_embed(tokens)
        for i, block in enumerate(self.blocks):
            x = block(x)
        x = self.ln_final(x)
        x = self.unembed(x)
        return x

    def training_step(
        self, batch: Dict[str, Tensor], batch_idx: int
    ) -> Float[Tensor, ""]:
        """
        Here you compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.
        """
        tokens = batch["tokens"].to(device)
        logits = self(tokens)
        loss = -get_log_probs(logits, tokens).mean()
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        return optimizer


# %%
# dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train").remove_columns(
#     "meta"
# )
dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")
model = ToyTransformer(cfg, args)
# model = model.to(device)
# compiled_model = torch.compile(model)
compiled_model = model
# tokenizer = ByteLevelBPETokenizer(vocab = "tiny_stories_10k_tokenizer-vocab.json", merges = "tiny_stories_10k_tokenizer-merges.txt")
# tokenizer = AutoTokenizer.from_pretrained("./tiny_tokenizer/")
# tokenizer_neo = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer = GPT2TokenizerFast(
    vocab_file="./tiny_tokenizer/tiny_stories_10k_tokenizer-vocab.json",
    merges_file="./tiny_tokenizer/tiny_stories_10k_tokenizer-merges.txt",
)

# %%
if os.path.exists(TOKENIZED_DATASET_PATH):
    tokenized_dataset = datasets.load_from_disk(TOKENIZED_DATASET_PATH)
else:
    tokenized_dataset = tokenize_and_concatenate(
        dataset,
        tokenizer,
        streaming=False,
        max_length=model.cfg.n_ctx,
        column_name="text",
        add_bos_token=True,
        num_proc=num_workers,
    )
    tokenized_dataset.save_to_disk(TOKENIZED_DATASET_PATH)


# %%
data_loader = DataLoader(
    tokenized_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)

# %%

# %%
trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    log_every_n_steps=args.log_every_n_steps,
    precision=precision,
    **additional_training_kwargs
)
trainer.fit(model=model, train_dataloaders=data_loader)  # , ckpt_path=checkpoint_path)
# %%
