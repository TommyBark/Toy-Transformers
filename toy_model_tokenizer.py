# %%
from pathlib import Path
import datasets
from tokenizers import ByteLevelBPETokenizer

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()
dataset = datasets.load_dataset("roneneldan/TinyStories", split="validation")
# Customize training
tokenizer.train_from_iterator(
    iterator=dataset["text"],
    vocab_size=10000,
    min_frequency=2,
    special_tokens=[
        "<PAD>",
        "<|endoftext|>",
    ],
)

# Save files to disk
tokenizer.save_model(".", "./tiny_tokenizer/tiny_stories_10k_tokenizer")

# %%
