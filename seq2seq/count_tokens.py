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
        model_args, data_args, data_training_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    elif len(sys.argv) == 3 and sys.argv[1].startswith("--local_rank") and sys.argv[2].endswith(".json"):
        data = json.loads(Path(os.path.abspath(sys.argv[2])).read_text())
        data.update({"local_rank": int(sys.argv[1].split("=")[1])})
        model_args, data_args, data_training_args, training_args = parser.parse_dict(args=data)
    else:
        model_args, data_args, data_training_args, training_args = parser.parse_args_into_dataclasses()
    

    # Set the flag to collect token counts
    print(data_training_args.collect_token_counts)


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

    # Collect results
    results = []
    for example in dataset_splits.train_split.dataset:  # or eval_split/test_splits as needed
        results.append(example)

    print(f"If you don't see output, make sure to delete your datataset cache at `.cache/spider_ssc_'lang'`")
    print(f"Saved token counts for {len(results)} samples")

if __name__ == "__main__":
    main()
