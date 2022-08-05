# entailment_bank

This is evaluation code for our recent paper **Explaining Answers with Entailment Trees**, EMNLP 2021 (https://arxiv.org/abs/2104.08661) 

Dataset available at https://allenai.org/data/entailmentbank

This dataset is also published in a more readable format as aprt of this book: http://cognitiveai.org/dist/entailmentbank-book-may2022.pdf

EntailmentBank annotation tool can be found at https://github.com/cognitiveailab/entailmentbank-tree-annotation-tool

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
python eval/run_scorer_task3.py \
  --split test \
  --prediction_file PREDICTION-TSV-PATH  \
  --output_dir  OUTPUT-PATH  \
  --bleurt_checkpoint "PATH to bleurt-large-512 model" 
  ```

### Prediction file format ###
Prediction file (PREDICTION-TSV-PATH) is a single column TSV file with datapoints in the same order as the public dataset jsonl file. Value of each line is the predicted proof in the DOT format.
e.g.
```
$proof$ = sent2 & sent3 -> int1: the northern hemisphere is a kind of place; int1 & sent1 -> hypothesis;
```

# Citation
```
@article{entalmentbank2021,
  title={Explaining Answers with Entailment Trees},
  author={Dalvi, Bhavana and Jansen, Peter and Tafjord, Oyvind and Xie, Zhengnan and Smith, Hannah and Pipatanangkura, Leighanna and Clark, Peter},
  journal={EMNLP},
  year={2021}
}
```

