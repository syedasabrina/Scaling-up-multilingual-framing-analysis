# Scaling-up-multilingual-framing-analysis

## Paper:
to_be_added

## Datasets: 
Due to the size limit and the fact that they are google translations of the english MFC and SNFC, the multilingual MFC and SNFC datasets couldn't be uploaded.


## Code:
For finetuning bert-based models,

```
python -u text_classification.py  \
      --epochs 2   --lr 1e-5 --batch_size 16 --output_file True --model 'roberta' \
      --train_path 'data/crowdsourced_data/SNFC_en.csv'  \
      --test_path 'data/MFC/MFC_test_en.csv'  --output_filename 'put-your-filename'
```

For LLM Prompting,

```
python prompting_mistral.py \
--learning_type 'few_shot'  --input_file_path 'data/data_for_prompting.csv' \
--out_file_name 'put_your_filename_here'

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
