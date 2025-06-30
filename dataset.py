import itertools
import random
import torch
from torch.utils.data import Dataset

SEQ_PAD_IX = 1
UNK_IX = 0

class AnnotatedCorpusDataset(Dataset):
    def __init__(self, examples, num_tags: int, input_vocab_size: int, ix_to_tag, tag_to_ix, ix_to_char, char_to_ix):
        self.num_tags = num_tags
        self.examples = examples
        self.input_vocab_size = input_vocab_size
        self.ix_to_tag = ix_to_tag
        self.tag_to_ix = tag_to_ix
        self.ix_to_char = ix_to_char
        self.char_to_ix = char_to_ix

    @staticmethod
    def _load_orig_dataset_raw(path):
        unique_tags = set()
        examples = []
        with open(path) as f:
            for line in f:
                # Skip - this is just filler
                if not line or line.startswith("<LINE#"):
                    continue

                raw_word, _morph_analysis, stem, pos = line.split()

                if pos == "V":
                    raw_word = stem

                raw_word = raw_word.lower()

                examples.append([raw_word, pos])
                unique_tags.add(pos)

        return AnnotatedCorpusDataset(examples, -1, -1, [], [], [], [])


    @staticmethod
    def _load_one_dataset_raw(path):
        unique_tags = set()
        examples = []
        with open(path) as f:
            for line in f:
                # Skip - this is just filler
                if not line or line.startswith("<LINE#"):
                    continue

                split = line.split()
                raw_word, pos = split[0], split[-1]
                examples.append([raw_word, pos])
                unique_tags.add(pos)


        return AnnotatedCorpusDataset(examples, -1, -1, [], [], [], [])

    @staticmethod
    def preprocess_split_data(in_path, train_path, dev_path):
        data = AnnotatedCorpusDataset._load_orig_dataset_raw(in_path)
        examples = data.examples

        examples_dedup = dict()
        for example in examples:
            if example[0] not in examples_dedup:
                examples_dedup[example[0]] = example[1]

        examples = list(examples_dedup.items())
        random.shuffle(examples)

        train = AnnotatedCorpusDataset(examples[len(examples) // 10:], data.num_tags, -1, [], [], [], [])
        dev = AnnotatedCorpusDataset(examples[:len(examples) // 10], data.num_tags, -1, [], [], [], [])

        for dataset, path in ((train, train_path), (dev, dev_path)):
            with open(path, "w") as f:
                f.write("\n".join(f"{ex[0]}\t{ex[1]}" for ex in dataset.examples))

    @staticmethod
    def load_data(device):
        train = AnnotatedCorpusDataset._load_one_dataset_raw("data/train.tsv")
        dev = AnnotatedCorpusDataset._load_one_dataset_raw("data/dev.tsv")

        tag_vocab = []
        in_vocab = [SEQ_PAD_IX, UNK_IX]

        for word, tag in itertools.chain(train.examples, dev.examples):
            for letter in word:
                if letter not in in_vocab:
                    in_vocab.append(letter)

            if tag not in tag_vocab:
                tag_vocab.append(tag)

        ix_to_tag = {i : tag for i, tag in enumerate(tag_vocab) }
        tag_to_ix = {tag : i for i, tag in enumerate(tag_vocab) }
        ix_to_char = {i : tag for i, tag in enumerate(in_vocab) }
        char_to_ix = {tag : i for i, tag in enumerate(in_vocab) }

        train.input_vocab_size = len(in_vocab)
        dev.input_vocab_size = len(in_vocab)
        train.num_tags = len(tag_vocab)
        dev.num_tags = len(tag_vocab)
        train.ix_to_tag = ix_to_tag
        train.tag_to_ix = tag_to_ix
        dev.ix_to_tag = ix_to_tag
        dev.tag_to_ix = tag_to_ix
        train.ix_to_char = ix_to_char
        train.char_to_ix = char_to_ix
        dev.ix_to_char = ix_to_char
        dev.char_to_ix = char_to_ix

        for examples in (dev.examples, train.examples):
            for i in range(len(examples)):
                word, tag = examples[i]

                examples[i][0] = torch.tensor([in_vocab.index(letter) for letter in word], device=device)
                examples[i][1] = torch.tensor(tag_vocab.index(tag), device=device)

        return train, dev

    def __getitem__(self, item):
        return self.examples[item]

    def __len__(self):
        return len(self.examples)

if __name__ == "__main__":
    AnnotatedCorpusDataset.preprocess_split_data(
        "data/SADII.XH.Morph_Lemma_POS.1.0.0.TRAIN.CTexT.TG.2021-09-30.txt",
        "data/train.tsv",
        "data/dev.tsv"
    )
