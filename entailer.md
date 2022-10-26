# Entailer model

The Entailer model is described in paper [Entailer: Answering Questions with Faithful and Truthful Chains of
Reasoning](https://www.semanticscholar.org/paper/Entailer%3A-Answering-Questions-with-Faithful-and-of-Tafjord-Dalvi/d400a649f0f0a3de22b89a268f48aff2dcb06a09) 
(EMNLP 2022). 

The core model is available in two sizes from the Hugging Face Model Hub: [entailer-large](https://huggingface.co/allenai/entailer-large) 
and [entailer-11b](https://huggingface.co/allenai/entailer-11b).

The model is a text-to-text transformer (built on top of [T5](https://github.com/google-research/text-to-text-transfer-transformer)), 
with certain conventions for how to format the input/output text.

The included [`nlp_agent.py`](utils/nlp_agent.py) utility file has some convenience functions for this, 
here is a simple example using entailer-large (for entailer-11b GPUs with at least 48GB total RAM 
should be used, e.g., for two GPUs can specify `cuda_devices=[0,1]` below):

```
# Load the model
from utils.nlp_agent import MultiAngleModel, NlpAgent
ew_model = MultiAngleModel(model_path="allenai/entailer-large", cuda_devices=None)
prover = NlpAgent(model=ew_model, default_outputs="proof")
entail_verifier = NlpAgent(model=ew_model, default_outputs=["implied"], default_options={"explicit_outputs": ['true', 'false']})
hyp_verifier = NlpAgent(model=ew_model, default_outputs=["valid"], default_options={"explicit_outputs": ['true', 'false']})

# Try to prove a hypothesis
hyp = "a magnet will not attract a penny"
proof = prover({"hypothesis": hyp})
premises = [x.strip() for x in proof.split("[PREMISE]") if x.strip()]

>>> proof
'[PREMISE] A magnet will not attract nonmagnetic materials. [PREMISE] A penny is always nonmagnetic.'
>>> premises
['A magnet will not attract nonmagnetic materials.', 'A penny is always nonmagnetic.']

# Does the model think the reasoning is good? Yes:
>>> entail_verifier({"hypothesis": hyp, "proof": proof})
{'implied': 'true', 'output_prob': 0.9999831914921629}

# Does the model believe the original hypothesis? No:
>>> hyp_verifier({"hypothesis": hyp})
{'valid': 'false', 'output_prob': 0.9711676239967346}

# Does the model believe the premises in the proof? Yes:
>>> hyp_verifier({"hypothesis": premises[0]})
{'valid': 'true', 'output_prob': 0.9990471005439758}
>>> hyp_verifier({"hypothesis": premises[1]})
{'valid': 'true', 'output_prob': 0.9997676014900208}
```

You can also force the first part of the output to specify, say, the first premise:

```
>>> proof_prefix = "[PREMISE] A penny is made of copper."
>>> prover({"hypothesis": hyp}, options={"output_prefix": {"proof": proof_prefix}})
'[PREMISE] A penny is made of copper. [PREMISE] A magnet will not attract copper.'
```

The input formats for the prover/hyp_verifier angles should be one of (the order of input fields matters):

```
{"hypothesis": hyp}
{"hypothesis": hyp, "context": c}
{"question": q, "answer": a, "hypothesis": hyp}
{"question": q, "answer": a, "hypothesis": hyp, "context": c}
```

where the optional context should look like 
"[HIGH] \<high sentences\> [MEDIUM] \<medium sentences\> [LOW] \<low sentences\>" as described in the appendix of the paper. 
For entail_verifier the "proof" field always comes right after "hypothesis".

## Citation

```
@article{entailer2022,
  title={Entailer: Answering Questions with Faithful and Truthful Chains of Reasoning},
  author={Tafjord, Oyvind and Dalvi, Bhavana and Clark, Peter},
  journal={EMNLP},
  year={2022}
}
```
