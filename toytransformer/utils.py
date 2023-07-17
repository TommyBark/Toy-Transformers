import nltk
import datasets
from typing import List, Union


def word_count(string: str) -> int:
    if string == " ":
        return 0
    return string.count(" ") + 1


def remove_sentences_with_characters(
    strings: List[str], characters: Union[List, str]
) -> List[str]:
    result = []
    for string in strings:
        if all(char not in string for char in characters):
            result.append(string)
    return result


def remove_punctuation(test_str: str) -> str:
    # Using filter() and lambda function to filter out punctuation characters
    result = "".join(
        filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), test_str)
    )
    return result


class SimpleTokenizer:
    def __init__(self):
        self.unique_words = {}
        self.tokenized_sentences = []

    def encode(self, sentences: List[str]) -> List[List[int]]:
        for sentence in sentences:
            words = sentence.split()
            tokenized_sentence = []

            for word in words:
                if word not in self.unique_words:
                    self.unique_words[word] = len(self.unique_words)
                tokenized_sentence.append(self.unique_words[word])

            self.tokenized_sentences.append(tokenized_sentence)

        return self.tokenized_sentences

    def decode(self, tokenized_sentences: List[List[int]]) -> List[str]:
        decoded_sentences = []
        reversed_unique_words = {
            index: word for word, index in self.unique_words.items()
        }

        for tokenized_sentence in tokenized_sentences:
            words = [reversed_unique_words[index] for index in tokenized_sentence]
            decoded_sentence = " ".join(words)
            decoded_sentences.append(decoded_sentence)

        return decoded_sentences


nltk.download("punkt")


def subset_dataset(
    dataset: datasets.arrow_dataset.Dataset, max_sentence_len: int = 6
) -> List[str]:
    """
    Subsets the large dataset to only sentences up to `max_sentence_len` words, for now it returns list of sentences instead of new dataset object.
    """
    sentences = []
    for paragraph in dataset["text"]:
        sentences.extend(nltk.sent_tokenize(paragraph))

    sentences = [sentence.lower().strip().strip(".") for sentence in sentences]
    sentences = remove_sentences_with_characters(
        sentences,
        [
            "\n",
            "$",
            "%",
            "<",
            "[",
            "]",
            "(",
            ")",
            "§",
        ],
    )
    sentences = [remove_punctuation(sentence) for sentence in sentences]

    ## fixing encoding
    fixed_sentences = []
    for s in sentences:
        try:
            fixed_string = s.encode("latin-1").decode("utf-16")
        except (UnicodeDecodeError, UnicodeEncodeError):
            fixed_string = s
        fixed_sentences.append(fixed_string)

    sentences = [
        sentence
        for sentence in fixed_sentences
        if (1 < word_count(sentence) <= max_sentence_len)
    ]
    return sentences


def get_n_params(model) -> int:
    """
    Returns number of model's trainable parameters
    """
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp
