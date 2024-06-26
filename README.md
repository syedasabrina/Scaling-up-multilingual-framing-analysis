# Scaling-up-multilingual-framing-analysis

## Paper:
https://aclanthology.org/2024.findings-naacl.260/

## Setup:
In a python venv, run:

```
pip install -r requirements.txt
```

## Code:
For finetuning bert-based models,

```
python -u text_classification.py  \
      --epochs 2   --lr 1e-5 --batch_size 16 --output_file True --model 'roberta' \
      --train_path $training_data_path  \
      --test_path $test_data_path  --output_filename $prediction_filename
```

For LLM Prompting,

```
python prompting_mistral.py \
--learning_type 'few_shot/zero_shot'  --input_file_path $prompting_data_file \
--out_file_name $prediction_filename

```

## Citation:

```
@inproceedings{akter-anastasopoulos-2024-framing,
    title = "A Study on Scaling up Multilingual Framing Analysis",
    author = "Akter, Syeda Sabrina and Anastasopoulos, Antonios",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "",
    doi = ""
}

```
