# %%
import os
import gc
from jaxtyping import Float, Int
from typing import List, Optional
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
    LitTransformer,
)
from transformer_lens.utils import tokenize_and_concatenate
from toytransformer.utils import get_n_params
from torch.utils.data import DataLoader
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    GPT2TokenizerFast,
)
from tqdm import tqdm
import pytorch_lightning as pl
from tokenizers import ByteLevelBPETokenizer
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph()


TOKENIZED_DATASET_PATH = "tokenized_dataset.hf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = len([torch.cuda.device(i) for i in range(torch.cuda.device_count())])
n_cpus:int = os.cpu_count()

if n_gpus > 1:
    additional_training_kwargs = {"accelerator":"gpu",
    "devices": n_gpus,
    "strategy":"ddp_find_unused_parameters_true"}
else:
    additional_training_kwargs = {
    }

if n_cpus > 8:
    num_workers = n_cpus - 2
else:
    num_workers = 0 

if device ==  torch.device("cuda"):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(False)
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.reset()
    torch._dynamo.config.verbose=True
    torch.set_float32_matmul_precision('medium')
    precision = 16
    #torch.cuda.empty_cache()
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
    batch_size = 4096*2
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
class ToyTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList(
            [TransformerBlock_noMLP(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        x = self.embed(tokens) + self.pos_embed(tokens)
        for i, block in enumerate(self.blocks):
            x = block(x)
        x = self.ln_final(x)
        x = self.unembed(x)
        return x


# dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train").remove_columns(
#     "meta"
# )
dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")
# %%
model = ToyTransformer(cfg)
#model = model.to(device)
compiled_model = torch.compile(model)
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
# lit_model = LitTransformer(args, compiled_model, data_loader)
lit_model = LitTransformer.load_from_checkpoint("/root/Toy-Transformers/lightning_logs/version_4/checkpoints/epoch=2-step=9774.ckpt", args=args, model = compiled_model, data_loader=data_loader)
trainer = pl.Trainer(
    max_epochs=args.max_epochs,
    log_every_n_steps=args.log_every_n_steps,
    precision="32-true",
**additional_training_kwargs
)
trainer.fit(model=lit_model, train_dataloaders=lit_model.data_loader)
# %%
