import sys
import os
import json
from pathlib import Path
from dataclasses import asdict
from transformers.hf_argparser import HfArgumentParser
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.models.auto import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from tokenizers import AddedToken

# project utils
from seq2seq.utils.args import ModelArguments
from seq2seq.utils.dataset import DataTrainingArguments, DataArguments
from seq2seq.utils.dataset_loader import load_dataset

def main():
    # Parse args (same way as your full script)
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    model_args: ModelArguments
    data_args: DataArguments
    data_training_args: DataTrainingArguments
    training_args: Seq2SeqTrainingArguments
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        try:
            model_args, data_args, data_training_args, training_args = parser.parse_json_file(
                json_file=os.path.abspath(sys.argv[1]),
                allow_extra_keys=True,
            )
        except TypeError:
            model_args, data_args, data_training_args, training_args = parser.parse_json_file(
                json_file=os.path.abspath(sys.argv[1])
            )
    elif len(sys.argv) == 3 and sys.argv[1].startswith("--local_rank") and sys.argv[2].endswith(".json"):
        data = json.loads(Path(os.path.abspath(sys.argv[2])).read_text())
        data.update({"local_rank": int(sys.argv[1].split("=")[1])})
        try:
            model_args, data_args, data_training_args, training_args = parser.parse_dict(args=data, allow_extra_keys=True)
        except TypeError:
            model_args, data_args, data_training_args, training_args = parser.parse_dict(args=data)
    else:
        model_args, data_args, data_training_args, training_args = parser.parse_args_into_dataclasses()
    

    # Token counting is implemented as a side-effect during preprocessing (see spider_pre_process_function).
    # This script just forces the relevant split(s) to be prepared/iterated.
    print(f"collect_token_counts={data_training_args.collect_token_counts}")

    # This script is not a trainer; we use the standard `do_*` flags only to select which split to preprocess.
    # Prefer test split when requested, otherwise validation, otherwise train.
    if training_args.do_predict:
        training_args.do_train = False
        training_args.do_eval = False
    elif training_args.do_eval:
        training_args.do_train = False
        training_args.do_predict = False
    elif training_args.do_train:
        training_args.do_eval = False
        training_args.do_predict = False
    else:
        raise ValueError(
            "Nothing to do: set at least one of `do_train`, `do_eval`, or `do_predict` "
            "to choose which split to count tokens for."
        )


    # Init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    assert isinstance(tokenizer, PreTrainedTokenizerFast), "Only fast tokenizers are supported"
    if isinstance(tokenizer, T5TokenizerFast):
        specials = [" {", " }", " <=", " <", "^^"]
        tokenizer.add_tokens([AddedToken(tok, normalized=True) for tok in specials])

    # Load dataset splits
    _, dataset_splits = load_dataset(
        data_args=data_args,
        model_args=model_args,
        data_training_args=data_training_args,
        training_args=training_args,   # not needed here
        tokenizer=tokenizer,
    )

    splits_to_iterate = []
    if training_args.do_predict:
        if dataset_splits.test_splits is None:
            raise ValueError("`do_predict` was set but no test splits were prepared.")
        for section, split in dataset_splits.test_splits.items():
            splits_to_iterate.append((section, split.dataset))
    elif training_args.do_eval:
        if dataset_splits.eval_split is None:
            raise ValueError("`do_eval` was set but no validation split was prepared.")
        splits_to_iterate.append(("validation", dataset_splits.eval_split.dataset))
    else:
        if dataset_splits.train_split is None:
            raise ValueError("`do_train` was set but no train split was prepared.")
        splits_to_iterate.append(("train", dataset_splits.train_split.dataset))

    total = 0
    for split_name, split_dataset in splits_to_iterate:
        # Iterate once to ensure preprocessing side-effects run even when caching is odd.
        n = 0
        for _ in split_dataset:
            n += 1
        total += n
        print(f"Prepared split `{split_name}` with {n} samples")

    print("If you don't see output, make sure to delete your dataset cache at `.cache/spider_ssc_*`.")
    print(f"Saved token counts for {total} samples (across {len(splits_to_iterate)} split(s))")

if __name__ == "__main__":
    main()
