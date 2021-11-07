# entailment_bank

## Setting up python environment
* conda create -n entbank python=3.7
* conda activate entbank
* pip install -r requirements.txt
* Download the bleurt-large-512 model from https://github.com/google-research/bleurt/blob/master/checkpoints.md

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

Task3:
```
example command coming up soon
```

### Prediction file format ###
Prediction file (PREDICTION-TSV-PATH) is a single column TSV file with datapoints in the same order as the public dataset jsonl file. Value of each line is the predicted proof in the DOT format.
e.g.
```
$proof$ = sent2 & sent3 -> int1: the northern hemisphere is a kind of place; int1 & sent1 -> hypothesis;
```

