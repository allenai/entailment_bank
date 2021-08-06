# entailment_bank

## Create a python environment and install requirements from requirements.txt
* conda create -n entbank python=3.7
* conda activate entbank
* pip install -r requirements.txt


## Example Evaluation Commands:

Task1:

```
python eval/run_scorer.py \
  --task "task_1" \
  --split test \
  --prediction_file PREDICTION-TSV-PATH  \
  --output_dir  OUTPUT-PATH  \
  --bleurt_checkpoint "PATH to bleurt-large-512 model"
```

Task2:
```
python eval/run_scorer.py \
  --task "task_2" \
  --split test \
  --prediction_file PREDICTION-TSV-PATH  \
  --output_dir  OUTPUT-PATH  \
  --bleurt_checkpoint "PATH to bleurt-large-512 model" 
```
