from typing import Dict
import json
import logging
import pickle
import os
import random
import re

from transformers import BartTokenizer, T5Tokenizer

logger = logging.getLogger(__name__)

DEFAULT_SLOT_FORMAT = {
    "slot": "$SLOT$",
    "assign": " = ",
    "separator": " ; ",
    "missing_value": "N/A"
}

SLOT_SHORTFORMS = {"Q": "question", "C": "context", "A": "answer", "E": "explanation",
                   "M": "mcoptions", "R": "rationale", "P": "proof",
                   "O": "original_question",
                   "H": "hypothesis",
                   "F": "full_text_proof",
                   "V": "valid"
                   }


def save_jsonl(file_name, data):
    with open(file_name, 'w') as file:
        for d in data:
            file.write(json.dumps(d))
            file.write("\n")


def load_jsonl(file_name):
    with open(file_name, 'r') as file:
        return [json.loads(line.strip()) for line in file]


def save_json(file_name, data):
    with open(file_name, 'w') as file:
        file.write(json.dumps(data))


### From https://github.com/huggingface/transformers/blob/master/examples/rag/utils.py

def encode_line(tokenizer, line, max_length, padding_side, pad_to_max_length=True, return_tensors="pt"):
    extra_kw = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) and not line.startswith(" ") else {}
    tokenizer.padding_side = padding_side
    return tokenizer(
        [line],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        add_special_tokens=True,
        **extra_kw,
    )


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def scramble_order(data, keep_last=None):
    keep_last = keep_last or []
    last = []
    other = []
    for d in data:
        if d in keep_last:
            last.append(d)
        else:
            other.append(d)
    random.shuffle(other)
    return other + last


def scramble_context_sentences(sentences, random_seed=137):
    scrambled_sentence_dict = dict()
    scrambled_sentence_list = []
    old_to_new_id_map = dict()
    sent_ids_orig = list(sentences.keys())
    random.shuffle(sent_ids_orig)
    for idx, sent_id_orig in enumerate(sent_ids_orig, start=1):
        new_sent_id = "sent" + str(idx)
        old_to_new_id_map[sent_id_orig] = new_sent_id
        scrambled_sentence_dict[new_sent_id] = sentences[sent_id_orig]
        scrambled_sentence_list.append(sentences[sent_id_orig])
    return scrambled_sentence_dict, scrambled_sentence_list, old_to_new_id_map


# Turns [['context', 'question','mcoptions'],['explanation', 'answer]] into 'CMQ->AE'
def shortform_angle(angle_full, sort_angle=True, overrides=None):
    if angle_full is None:
        return ""   
    if sort_angle:
        return "->".join(["".join(sorted([angle[0].upper() for angle in angles])) for angles in angle_full])
    return "->".join(["".join([angle[0].upper() for angle in angles]) for angles in angle_full])

def decompose_slots(string, fmt=None):
    fmt = fmt or DEFAULT_SLOT_FORMAT
    string = string.strip()
    no_slot = "PREFIX"
    slot_re = re.compile('(?i)'+re.escape(fmt['slot']).replace("SLOT", "(\\w*?)"))
    assign_re = re.escape(fmt['assign']).replace('\\ ','\\s*')
    separator_re = re.escape(fmt['separator']).replace('\\ ','\\s*')
    strip_re = re.compile(f"^({assign_re})?(.*?)({separator_re})?$")
    slot_pos = []
    for m in slot_re.finditer(string):
        slot_pos.append((m.span(), m.group(1)))
    if len(slot_pos) == 0:
        return {no_slot: string}
    if slot_pos[0][0][0] > 0:
        slot_pos = [((0,-1), no_slot)] + slot_pos
    res = {}
    for idx, (pos, slot_name) in enumerate(slot_pos):
        if idx == len(slot_pos) - 1:
            value = string[pos[1]+1:]
        else:
            value = string[pos[1]+1:slot_pos[idx+1][0][0]-1]
        m = strip_re.match(value)
        if m is not None:
            value = m.group(2)
        value = value.strip()
        if slot_name in res:
            value = res[slot_name] + " ~AND~ " + value
        res[slot_name] = value
    return res


def slot_file_to_angles(slot_file, slot_shortforms,
                        angle_distribution,
                        split,
                        full_train_first_angle=False,
                        meta_fields=None,
                        id_filter_regex=None,
                        train_replicas=1,
                        random_seed=137, **kwparams):

    res = []
    random.seed(random_seed)
    if split == 'train':
        if full_train_first_angle:
            angle_distributions = [angle_distribution[0][0]] + [angle_distribution] * train_replicas
        else:
            angle_distributions = [angle_distribution] * train_replicas
    else:
        angle_distributions = angle_distribution[0]
    with open(slot_file, 'r') as file:
        for line in file:
            fields = json.loads(line.strip())
            if id_filter_regex is not None and "id" in fields and not re.match(id_filter_regex, fields['id']):
                continue
            slot_data = SlotDataInstance(fields)
            for ad in angle_distributions:
                instance = slot_data.sample_angle_instance(ad, slot_shortforms, **kwparams)
                instance.update({"id": fields.get('id', 'NA')})
                res.append(instance)
                if meta_fields:
                    instance['meta']  = {x:fields['meta'][x] for x in meta_fields}
    return res


ANGLE_SPEC_DEFAULT = {'angle_distribution': None,
                      'full_train_first_angle': False,
                      'id_filter_regex': None,
                      'train_replicas': 1,
                      'meta_fields': [],
                      'random_seed': 137,
                      'keep_last': ['context'],
                      'scramble_slots': True,
                      'multi_value_sampling': None
                      }


def build_angle_dir(slot_dir, angle_dir, angle_spec, debug_print=2):
    if os.path.exists(angle_dir):
        raise ValueError(f"Angle data directory {angle_dir} already exist!")
    os.makedirs(angle_dir)
    angle_spec = {**ANGLE_SPEC_DEFAULT, **angle_spec}
    angle_spec_file = os.path.join(angle_dir, "angle_spec.json")

    save_json(angle_spec_file, angle_spec)
    made_splits = []
    for split in ['train', 'dev', 'val', 'test']:
        slot_file = os.path.join(slot_dir, split+".jsonl")
        if os.path.exists(slot_file):
            logger.info(f"Creating angle data for {slot_file}")
            angle_data = slot_file_to_angles(slot_file, SLOT_SHORTFORMS, split=split, **angle_spec)
            if debug_print > 0:
                logger.info(f"Sample angle data: {angle_data[:debug_print]}")
            angle_file = os.path.join(angle_dir, split+".jsonl")
            save_jsonl(angle_file, angle_data)
            made_splits.append((split, len(angle_data)))
    logger.info(f"Created angle data for splits {made_splits}.")
    return made_splits


def save_tsv_file(file_name, data):
    with open(file_name, "w") as f:
        for d in data:
            out = "\t".join([s.replace('\n', ' ').replace('\t', ' ') for s in d])
            f.write(out + '\n')

# Use small_dev = 2000 to save a smaller size-2000 dev set
def convert_angle_dir_tsv(angle_dir, tsv_dir, small_dev=False):
    if os.path.exists(tsv_dir):
        raise ValueError(f"TSV data directory {tsv_dir} already exist!")
    os.makedirs(tsv_dir)
    counts = {}
    for split in ['train', 'dev', 'val', 'test']:
        angle_file = os.path.join(angle_dir, split+".jsonl")
        if os.path.exists(angle_file):
            logger.info(f"Creating tsv data for {angle_file}")
            angle_data = load_jsonl(angle_file)
            tsv_data = [[x['input'], x['output']] for x in angle_data]
            meta_data =[[x['id'], shortform_angle(x['angle'], sort_angle=False)] for x in angle_data]
            if small_dev and split in ['dev', 'val']:
                num_dev = small_dev if isinstance(small_dev, int) else 1000
                counts[split+"-full"] = len(tsv_data)
                save_tsv_file(os.path.join(tsv_dir, split+"-full.tsv"), tsv_data)
                save_tsv_file(os.path.join(tsv_dir, "meta-"+split+"-full.tsv"), meta_data)
                tsv_data = tsv_data[:num_dev]
                meta_data = meta_data[:num_dev]
            counts[split] = len(tsv_data)
            save_tsv_file(os.path.join(tsv_dir, split + ".tsv"), tsv_data)
            save_tsv_file(os.path.join(tsv_dir, "meta-" + split + ".tsv"), meta_data)
    save_json(os.path.join(tsv_dir, "counts.json"), counts)
    logger.info(f"Created angle data for splits {counts}.")
    return counts


class SlotDataInstance():

    def __init__(self, fields: Dict):
        self.fields = fields
        self.slot_value_sampling = {}

    def get_slot_value(self, slot, default=None, multi_value_sampling=None):
        res = self.fields.get(slot, default)
        if isinstance(res, list):
            if multi_value_sampling is not None and slot in multi_value_sampling:
                fn = multi_value_sampling[slot]
                if fn == "random":
                    res = random.choice(res)
                elif "random-with" in fn:
                    other_slots = fn.split("-")[2:]
                    value_index = -1
                    for other_slot in other_slots:
                        if other_slot in self.slot_value_sampling:
                            value_index = self.slot_value_sampling[other_slot]
                    if value_index == -1:
                        value_index = random.choice(range(len(res)))
                    self.slot_value_sampling[slot] = value_index
                    res = res[value_index]
                else:
                    raise ValueError(f"Unknown multi_value_sampling function {fn}")
            else:
                res = res[0]
        return res

    def convert_shortform_angle(self, angle, slot_shortforms):
        if isinstance(angle, str):
            arrowpos = angle.index("->")
            lhs = angle[:arrowpos].strip()
            rhs = angle[arrowpos + 2:].strip()
            lhs = [slot_shortforms[c] for c in lhs]
            rhs = [slot_shortforms[c] for c in rhs]
        else:
            lhs = angle[0]
            rhs = angle[1]
        missing = [slot for slot in lhs + rhs if slot not in self.fields]
        return ((lhs, rhs), missing)

    def make_angle_instance(self, angle, fmt=None, multi_value_sampling=None):
        fmt = fmt or DEFAULT_SLOT_FORMAT
        lhs = []
        rhs = []
        for slot in angle[1]:
            slot_name = fmt['slot'].replace("SLOT", slot)
            slot_value = self.get_slot_value(slot, fmt['missing_value'], multi_value_sampling)
            lhs.append(slot_name)
            rhs.append(f"{slot_name}{fmt['assign']}{slot_value}")
        for slot in angle[0]:
            slot_name = fmt['slot'].replace("SLOT", slot)
            slot_value = self.get_slot_value(slot, fmt['missing_value'], multi_value_sampling)
            lhs.append(f"{slot_name}{fmt['assign']}{slot_value}")
        return {"input": fmt['separator'].join(lhs),
                "output": fmt['separator'].join(rhs),
                "angle": angle}

    def sample_angle_instance(self, angle_distribution, slot_shortforms,
                              scramble_slots=True,
                              keep_last=None,
                              missing_retries=100,
                              fmt=None,
                              multi_value_sampling=None):
        keep_last = keep_last or ["context"]
        fmt = fmt or DEFAULT_SLOT_FORMAT
        if isinstance(angle_distribution, str):
            angle, missing = self.convert_shortform_angle(angle_distribution, slot_shortforms)
        else:
            angle_distribution = tuple(x.copy() for x in angle_distribution)
            retries = missing_retries
            missing = [1]
            while retries >= 0 and len(missing) > 0 and len(angle_distribution[0]) > 0:
                retries -= 1
                angle_shortform = random.choices(*angle_distribution)[0]
                angle, missing = self.convert_shortform_angle(angle_shortform, slot_shortforms)
                if len(missing) > 0:
                    ind = angle_distribution[0].index(angle_shortform)
                    angle_distribution[0].pop(ind)
                    angle_distribution[1].pop(ind)
        if scramble_slots:
            angle = [scramble_order(a, keep_last) for a in angle]
        res = self.make_angle_instance(angle, fmt, multi_value_sampling)
        return res


def formatting(a_set):
    return '=CONCATENATE("' + ('" ; CHAR(10); "').join(a_set) + '"' + ')' if len(a_set) > 0 else ""


def get_selected_str(data_dict, selected_keys, format=False):
    selected=[]
    for key in selected_keys:
        selected.append(f"{key}: {data_dict[key]['text']} ")
    if format:
        selected_str = formatting(selected)
    else:
        selected_str = ''.join(selected)
    return selected_str


def get_selected_keys(data_dict, selected_keys, format=False):
    selected=[]
    for key in selected_keys:
        selected.append(f"{key}: {data_dict[key]} ")
    if format:
        selected_str = formatting(selected)
    else:
        selected_str = ''.join(selected)
    return selected_str
