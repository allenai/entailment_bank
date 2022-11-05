# entailment_bank

This repository contains evaluation code for the 
paper [**Explaining Answers with Entailment Trees**](https://arxiv.org/abs/2104.08661) (EMNLP 2021), see below.

See [entailer.md](entailer.md) for information on how to run the Entailer model from follow-up paper: 
[Entailer: Answering Questions with Faithful and Truthful Chains of
Reasoning](https://www.semanticscholar.org/paper/Entailer%3A-Answering-Questions-with-Faithful-and-of-Tafjord-Dalvi/d400a649f0f0a3de22b89a268f48aff2dcb06a09) 
(EMNLP 2022).


## EntailmentBank evaluation code

Dataset available at https://allenai.org/data/entailmentbank

This dataset is also published in a more readable format as part of this book: http://cognitiveai.org/dist/entailmentbank-book-may2022.pdf

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

For task 3, when using a custom retrieved set of sentences, the prediction file should instead be a jsonl file 
where each line includes a `'worldtree_provenance'` field specifying what each sentence refers to, 
as well as an `'id'` and `'slots'` + `'proof'` field, following the format in this example:
```
{"id":"Mercury_SC_401371","slots":{"proof":"sent23 & sent3 -> int1: the sun rising and setting are kinds of events; int1 & sent4 -> hypothesis;"},
 "worldtree_provenance":{"sent12":{"uuid":"a972-0cbb-3c14-b098","original_text":"cycles of day and night occur once per day"},
   "sent4":{"uuid":"f8be-9407-d60e-6282","original_text":"the sun rising; setting occurs once per day"},...}}
```

# Citation
```
@article{entailmentbank2021,
  title={Explaining Answers with Entailment Trees},
  author={Dalvi, Bhavana and Jansen, Peter and Tafjord, Oyvind and Xie, Zhengnan and Smith, Hannah and Pipatanangkura, Leighanna and Clark, Peter},
  journal={EMNLP},
  year={2021}
}
```

