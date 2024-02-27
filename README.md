# Scaling-up-multilingual-framing-analysis

For finetuning,

python -u text_classification.py  \
      --epochs 2   --lr 1e-5 --batch_size 16 --output_file True --model 'roberta' \
      --train_path 'data/crowdsourced_data/SNFC_en.csv'  \
      --test_path 'data/MFC/MFC_test_en.csv'  --output_filename 'put-your-filename'
