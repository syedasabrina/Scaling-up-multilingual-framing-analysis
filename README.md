# Scaling-up-multilingual-framing-analysis


## Datasets: 
Due to the size limit and the fact that they are google translations of the english MFC and SNFC, the multilingual MFC and SNFC datasets couldn't be uploaded.

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
