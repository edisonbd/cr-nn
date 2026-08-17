"""Train a small BPE tokenizer (8K) on the formal corpus for meaningful ppl.

A 50K GPT-2 vocab cannot learn from 2M tokens (long tail of rare subwords),
so for small-data language modelling we fit a compact subword vocab that is
still fully subword (not char-level), letting the model actually generalise.
"""
import sys

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/formal_corpus.txt"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/bpe8k-tok"
    vocab = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab, special_tokens=["<unk>", "<s>", "</s>", "[MASK]"],
        min_frequency=2)
    tok.train([src], trainer)
    tok.save(f"{out}/tokenizer.json")
    # minimal config so transformers can load it
    import json
    cfg = {"unk_token": "<unk>", "model_max_length": 100000}
    with open(f"{out}/tokenizer_config.json", "w") as f:
        json.dump(cfg, f)
    print(f"saved {out} vocab={tok.get_vocab_size()}")


if __name__ == "__main__":
    main()
