import logging
import pandas as pd
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Union

from tqdm import tqdm
import datasets,argparse
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    default_data_collator
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


logger = logging.getLogger(__name__)

@dataclass
class DataCollatorForMultipleChoice:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    dataset_type:Optional[str] = None

    def __call__(self, features):
        label_name ="label"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Question Answering task")
    parser.add_argument(
        "--test_dataset",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_predict_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="cycic",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        default=False,
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--per_device_test_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    args = parser.parse_args()
    return args


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    args = parse_args()
    accelerator_log_kwargs = {}
    accelerator = Accelerator(gradient_accumulation_steps=2, **accelerator_log_kwargs)

    data_files = {}
    if args.test_dataset is not None:
        data_files["test"] = args.test_dataset
    extension = args.test_dataset.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    config = AutoConfig.from_pretrained(args.config_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = AutoModelForMultipleChoice.from_pretrained(
            args.model_path,
            use_safetensors=True,
            from_tf=bool(".ckpt" in args.model_path),
            config=config,
        )
    
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    def preprocess_function(examples):
        # When using your own dataset or a different dataset from swag, you will probably need to change this.
        if args.dataset_type=="cycic":
            answer_num=5
            ending_names = [f"answer_option{i}" for i in range(answer_num)]
            first_sentences = [[q] * answer_num for q in examples['question']]
            second_sentences = [
                [f"{examples[end][i]}" for end in ending_names] for i, _ in enumerate(examples['answer_option0'])
            ]
            labels = examples['correct_answer']
        else:
            answer_num=4
            ending_names = [f"choice_{i}" for i in range(answer_num)]
            first_sentences = [[q] * answer_num for q in examples['question']]
            second_sentences = [
                [f"{examples[end][i]}" for end in ending_names] for i, _ in enumerate(examples['choice_0'])
            ]
            labels = examples['answerKey']

        print(f"\n1st sentence(Question):{first_sentences[0]}\n2nd sentence(Choices):{second_sentences[0]}\nlabel:{labels[0]}")
        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length" if args.pad_to_max_length else False,
        )
        # Un-flatten
        tokenized_inputs = {k: [v[i : i + answer_num] for i in range(0, len(v), answer_num)] for k, v in tokenized_examples.items()}
        tokenized_inputs["label"] = labels
        return tokenized_inputs

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
                    preprocess_function, 
                    batched=True,
                    remove_columns=raw_datasets["test"].column_names
                )

    test_dataset = processed_datasets["test"]

    # Data collator
    data_collator = (
        default_data_collator
        if args.pad_to_max_length
        else DataCollatorForMultipleChoice(tokenizer=tokenizer, pad_to_multiple_of=None,dataset_type=args.dataset_type)
    )

    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_test_batch_size)
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    index,wrong=0,0
    model.eval()
    all_predict=[]
    print("\nStart for Multiple choices prediction:")
    for step, batch in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            outputs = model(**batch)
            predictions_per_batch = outputs.logits.argmax(dim=-1)
            for p in predictions_per_batch.tolist():
                all_predict.append(p)
                if args.dataset_type=="cycic":
                    true_ans=raw_datasets['test'][f'correct_answer'][index]
                    if int(true_ans)!=p:
                        wrong+=1
                        #print(f"\nwrong prediction for question {index}:choice_{p}({raw_datasets['test'][f'answer_option{p}'][index]})\ntrue answer:{true_ans}({raw_datasets['test'][f'answer_option{int(true_ans)}'][index]})")
                elif args.dataset_type=="ARC":
                    true_ans=raw_datasets['test'][f'answerKey'][index]
                    if int(true_ans)!=p:
                        wrong+=1
                        #print(f"\nwrong prediction for question {index}:choice_{p}({raw_datasets['test'][f'choice_{p}'][index]})\ntrue answer:{true_ans}({raw_datasets['test'][f'choice_{int(true_ans)}'][index]})")
                    
                index+=1

    print(f"question num:{index},wrong num:{wrong},accuracy:{round((1-wrong/index)*100,4)}%")
    test_pd=pd.read_csv(args.test_dataset)
    test_pd.insert(1 if args.dataset_type=="cycic" else 2, 'predict_answer', all_predict)
    test_pd.to_csv(args.output_predict_path,index=False)
                



if __name__ == "__main__":
    main()