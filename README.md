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
@inproceedings{akter-anastasopoulos-2024-study,
    title = "A Study on Scaling Up Multilingual News Framing Analysis",
    author = "Akter, Syeda Sabrina  and
      Anastasopoulos, Antonios",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Findings of the Association for Computational Linguistics: NAACL 2024",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-naacl.260",
    pages = "4156--4173",
    abstract = "Media framing is the study of strategically selecting and presenting specific aspects of political issues to shape public opinion. Despite its relevance to almost all societies around the world, research has been limited due to the lack of available datasets and other resources. This study explores the possibility of dataset creation through crowdsourcing, utilizing non-expert annotators to develop training corpora. We first extend framing analysis beyond English news to a multilingual context (12 typologically diverse languages) through automatic translation. We also present a novel benchmark in Bengali and Portuguese on the immigration and same-sex marriage domains.Additionally, we show that a system trained on our crowd-sourced dataset, combined with other existing ones, leads to a 5.32 percentage point increase from the baseline, showing that crowdsourcing is a viable option. Last, we study the performance of large language models (LLMs) for this task, finding that task-specific fine-tuning is a better approach than employing bigger non-specialized models.",
}

```
