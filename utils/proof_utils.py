import logging
import math
import os
import random
import re

from utils.angle_utils import load_jsonl

logger = logging.getLogger(__name__)

def parse_lisp(input_string, token_map=None):
    token_regex = re.compile("(->|[a-zA-Z0-9%@^]+|\\S)")
    tokens = token_regex.findall(input_string)
    if token_map is not None:
        tokens = [token_map.get(x, x) for x in tokens]
    return parse_lisp_tokens(tokens)


def parse_lisp_tokens(tokens):
    res = []
    i = 0
    while i < len(tokens):
        symbol = tokens[i]
        if symbol == '(':
            sub_list = []
            depth = 1
            while depth > 0:
                i += 1
                if i >= len(tokens): raise ValueError("Invalid input: Unmatched open parenthesis.")
                symbol = tokens[i]
                if symbol == '(': depth += 1
                elif symbol == ')': depth -= 1
                if depth > 0: sub_list.append(symbol)
            res.append(parse_lisp_tokens(sub_list))
        elif symbol == ')':
                raise ValueError("Invalid input: Unmatched close parenthesis.")
        else:
            res.append(symbol)
        i += 1
    return res


SYMBOL_DICT = {"->": "#", "and": "&"}  # we invert -> to <- implicitly as well


def recursive_polish_notation(proof_list):
    res = []
    if isinstance(proof_list, str):
        res.append(proof_list)
        return res
    if len(proof_list) == 3 and proof_list[1] == "->":
        res.append(SYMBOL_DICT['->'])
        res += recursive_polish_notation(proof_list[2])
        res += recursive_polish_notation(proof_list[0])
        return res
    # Remove duplicate arguments in and (not recursively though)
    proof_list = dedupe(proof_list)
    if len(proof_list) == 1:
        res += recursive_polish_notation(proof_list[0])
    elif len(proof_list) == 2:
        res.append(SYMBOL_DICT['and'])
        res += recursive_polish_notation(proof_list[0])
        res += recursive_polish_notation(proof_list[1])
    elif len(proof_list) > 2:
        res.append(SYMBOL_DICT['and'])
        res += recursive_polish_notation(proof_list[0])
        res += recursive_polish_notation(proof_list[1:])
    else:
        raise ValueError("Empty list found")
    return res


def proof_to_polish_notation(proof_string, token_map=None):
    proof_list = parse_lisp(proof_string, token_map)
    #print(f"proof_list:{proof_list}")
    if len(proof_list) != 1:
        print(f"Failed proof parse for {proof_string}  {token_map}")
    assert len(proof_list) == 1
    res = recursive_polish_notation(proof_list[0])
    return " ".join(res)


def combine_int_token(token, intermediate, token_map):
    if token_map is not None:
        token = token_map.get(token, token)
        intermediate = token_map.get(intermediate, intermediate)
    return f"{token}@{intermediate}"


def remap_tokens_in_order(proof, prefix):
    ints = re.findall(f"\\b{prefix}\\d+", proof)
    remap = {}
    for int in ints:
        if int not in remap:
            remap[int] = prefix + str(len(remap) + 1)
    new_proof = re.sub(f"\\b({prefix}\\d+)", lambda x: remap[x[1]], proof)
    return remap, new_proof


def proof_and_intermediates_to_polish_notation(proof_details, token_map=None, sort_ints=False):
    # collapse (rule4 % int3) to rule4%int3 single token)
    proof_string = re.sub("\\((rule\\d+)\\s*%\\s*(int\\d+)\\)",
                          lambda x: combine_int_token(x[1], x[2], token_map),
                          proof_details['representation'])
    proof_polish = proof_to_polish_notation(proof_string, token_map)
    tmp = proof_polish
    new_intermediates = proof_details['intermediates']
    if len(proof_details['intermediates']) == 0:
        intermediates = ""
    else:
        if sort_ints:
            remap, proof_polish = remap_tokens_in_order(proof_polish, "int")
            remap_naf, proof_polish = remap_tokens_in_order(proof_polish, "naf")
            remap.update(remap_naf)
            try:
                new_intermediates = {remap[k]: v for k, v in new_intermediates.items()}
            except:
                print(f"{proof_string}  \n{proof_polish} \n{tmp}  \n{remap}  \n{remap_naf}")
                print(1/0)
            new_intermediates = {k: new_intermediates[k] for k in sorted(new_intermediates.keys())}
        intermediates = [k + ": " + v['text'] for k, v in new_intermediates.items()]
        intermediates = " ; ".join(intermediates)
    return proof_polish, intermediates, new_intermediates


def recursive_from_polish_notation(tokens, num_args=1, start_index=0):
    res = []
    i = start_index
    while i < len(tokens):
        symbol = tokens[i]
        if symbol in SYMBOL_DICT.values():
            args, end_index = recursive_from_polish_notation(tokens, num_args=2, start_index=i + 1)
            if symbol == SYMBOL_DICT['->']:
                res.append([args[1], "->", args[0]])
            elif symbol == SYMBOL_DICT['and']:
                res.append(args)
            i = end_index
        else:
            res.append(symbol)
            i += 1
        if len(res) == num_args:
            return res, i
    raise ValueError("Not enough arguments found!")


def polish_notation_to_proof(polish_notation):
    tokens = polish_notation.split(" ")
    res, end_index = recursive_from_polish_notation(tokens)
    if end_index != len(tokens):
        raise ValueError("Full string was not processed!")
    return res


def get_proof_without_ints(proof):
    if not isinstance(proof, str):
        return proof
    proof_core = proof.split(";")[0].strip()  # remove "; with int1 = ..."
    proof_clean = re.sub("(naf|int)\\d+", "\\1", proof_core)
    proof_clean = re.sub(" ⁇ ", "^", proof_clean)  # hack around using UNK ^ token, bleurgg
    return proof_clean


def decouple_proof_struct_ints(proof):
    int_dict = dict()
    if not isinstance(proof, str):
        return proof
    proof_clean = proof.split(";")[0].strip()  # remove "; with int1 = ..."
    ints = proof.split("; with ")[1]
    print(f"ints:{ints}")
    for int in ints.split(';'):
        i_parts = int.strip().split(":")
        int_id = i_parts[0].strip()
        int_str = i_parts[1].strip()
        int_dict[int_id] = int_str

    # proof_clean = re.sub("(naf|int)\\d+", "\\1", proof_core)
    proof_clean = re.sub(" ⁇ ", "^", proof_clean)  # hack around using UNK ^ token, bleurgg
    return proof_clean, int_dict


def score_pn_proof(pn_proof, gold_pn_proofs):
    pn_proof = get_proof_without_ints(pn_proof)
    gold_pn_proofs = [get_proof_without_ints(p) for p in gold_pn_proofs]
    try:
        parsed_proof = polish_notation_to_proof(pn_proof)
    except:
        return {"acc": 0, "bad_parse": 1}
    try:
        gold_normalized = [normalize_proof(polish_notation_to_proof(p)) for p in gold_pn_proofs]
    except:
        raise ValueError(f"Bad gold proofs in {gold_pn_proofs}")
    pred_normalized = normalize_proof(parsed_proof)
    score = 1 if pred_normalized in gold_normalized else 0
    return {"acc": score}


def dedupe(data):
    res = []
    for d in data:
        if d not in res:
            res.append(d)
    return res


def normalize_proof(proof_list):
    if isinstance(proof_list, str):
        return proof_list
    if len(proof_list) == 3 and proof_list[1] == "->":
        return [normalize_proof(x) for x in proof_list]
    and_elements = []
    for elem in proof_list:
        norm = normalize_proof(elem)
        if isinstance(norm, str) or (len(norm) == 3 and norm[1] == "->"):
            and_elements.append(norm)
        else:
            and_elements += norm
    and_elements = dedupe(and_elements)  # Delete duplicates
    if len(and_elements) == 1:
        return and_elements[0]
    and_elements.sort(key=lambda x: str(x))
    return and_elements


NO_INFERENCE = "Nothing."
NO_PROOF = "None"

# Set proofs to include proof slot per question.
# Set align_data to map proofs to correct sentences for paraphrased text
def make_ruletaker_slots(meta, scramble_sentences=False, sentence_prefix="sent",
                         proofs="PN", include_intermediates=False, align_data=None):
    theory_id = meta['id']
    context = []
    token_map = {}
    if 'NatLang' in theory_id:
        if align_data is not None:
            align = align_data[theory_id]
        else:
            align = meta
        sent_ids_orig = list(align['sentences'].keys())
        if scramble_sentences:
            random.shuffle(sent_ids_orig)
        sentence_id_map = {}
        for idx, sent_id_orig in enumerate(sent_ids_orig):
            sent_id = sentence_prefix + str(idx+1)
            sentence_id_map[sent_id_orig] = sent_id
            sentence = align['sentences'][sent_id_orig]
            context.append(f"{sent_id}: {sentence}")
        for k, v in align['mappings'].items():
            token_map[k] = sentence_id_map[v]
    else:
        sentences = [(k, v['text']) for k, v in meta['triples'].items()] + \
                    [(k, v['text']) for k, v in meta['rules'].items()]
        if scramble_sentences:
            random.shuffle(sentences)
        for idx, sentence in enumerate(sentences):
            sent_id = sentence_prefix + str(idx+1)
            token_map[sentence[0]] = sent_id
            context.append(f"{sent_id}: {sentence[1]}")
    context_string = " ".join(context)
    res = []
    for q_id, question in meta['questions'].items():
        slots = {"id": f"{theory_id}-{q_id}"}
        slots['context'] = context_string
        slots['answer'] = str(question['answer'])
        q_string = question['question']
        if q_string.endswith("."):
            q_string = q_string[:-1]+"?"
        slots['question'] = q_string
        proof_intermediates = []
        proof_strings = re.split("\\s*\\bOR\\b\\s*", question['proofs'][2:-2])
        # Sort proofs by fewest implications, so shortest is gold proof
        proof_strings.sort(key=lambda x: x.count("->"))
        if proofs == "PN":
            if not 'proof' in question['strategy']:
                proof_slot = [NO_PROOF]
            else:
                if include_intermediates:
                    proof_full = question['proofsWithIntermediates']
                    proof_slot = []
                    proof_strings = []
                    proof_intermediates = []
                    for p in proof_full:
                        proof_pn, proof_int, new_ints = proof_and_intermediates_to_polish_notation(p, token_map, True)
                        proof_combined = proof_pn
                        proof_intermediates.append(new_ints)
                        if proof_int:
                            proof_combined += " ; with " + proof_int
                        proof_slot.append(proof_combined)
                        proof_strings.append(p['representation'])
                else:
                    proof_slot = [proof_to_polish_notation(p, token_map) for p in proof_strings]
                    for (pn, p) in zip(proof_slot, proof_strings):
                        assert normalize_proof(polish_notation_to_proof(pn)) == normalize_proof(parse_lisp(p, token_map))
            if len(proof_slot) == 1:
                proof_slot = proof_slot[0]
            slots['proof'] = proof_slot
        slots['meta'] = {"QDep": question['QDep'], "QLen": question['QLen'],
                         "strategy": question['strategy'], "proofs_orig": proof_strings,
                         "sentence_map": token_map}
        if include_intermediates:
            slots['meta']['proof_intermediates'] = proof_intermediates
        res.append(slots)
    return res


# Input: "allProofs" field from RuleTaker data
# Sample output: [{'depth': 0, 'assertion': 'Erin is big.', 'proof': '[(triple1)]'},
#                 {'depth': 1, 'assertion': 'Erin is round.', 'proof': '[(((NAF) -> rule1))]'},
#                 {'depth': 1, 'assertion': 'Erin is quiet.', 'proof': '[(((triple1) -> rule5))]'}]
def from_all_proofs_field(all_proofs):
    res = []
    for match in re.finditer("@(\\d+): *([^@]*)", all_proofs):
        depth = int(match.group(1))
        for match2 in re.finditer("([^[]*)(\\[.*?\\])", match.group(2)):
            proofs = re.split("\\s*\\bOR\\b\\s*", match2.group(2).strip()[2:-2])
            res.append({"depth": depth, "assertion": match2.group(1).strip(), "proofs": proofs})
    return res


def make_ruletaker_slots_all_inferences(meta, scramble_sentences=False, sentence_prefix="sent", proofs="PN",
                                        align_data=None, one_inference_per_answer=False,
                                        one_hop_inferences=False,
                                        filter_naf_inferences=False,
                                        inference_answer_as_list=False):
    theory_id = meta['id']
    context = []
    token_map = {}
    if 'NatLang' in theory_id:
        if align_data is not None:
            align = align_data[theory_id]
        else:
            align = meta
        sent_ids_orig = list(align['sentences'].keys())
        if scramble_sentences:
            random.shuffle(sent_ids_orig)
        sentence_id_map = {}
        for idx, sent_id_orig in enumerate(sent_ids_orig):
            sent_id = sentence_prefix + str(idx+1)
            sentence_id_map[sent_id_orig] = sent_id
            sentence = align['sentences'][sent_id_orig]
            context.append(f"{sent_id}: {sentence}")
        for k, v in align['mappings'].items():
            token_map[k] = sentence_id_map[v]
    else:
        rules = [(k, v['text']) for k, v in meta['rules'].items()]
        triples = [(k, v['text']) for k, v in meta['triples'].items()]
        if scramble_sentences == "rules_first":
            sentences = rules + triples
        elif scramble_sentences == "rules_first_random":
            random.shuffle(rules)
            random.shuffle(triples)
            sentences = rules + triples
        else:
            sentences = triples + rules
            if scramble_sentences is True:
                random.shuffle(sentences)
        for idx, sentence in enumerate(sentences):
            sent_id = sentence_prefix + str(idx+1)
            token_map[sentence[0]] = sent_id
            context.append(f"{sent_id}: {sentence[1]}")
    context_string = " ".join(context)
    question_text = "What are all the inferences?"
    if one_inference_per_answer:
        question_text = "What is one inference?"
    if one_hop_inferences:
        question_text = "What is one single-hop inference?"
        all_inferences = [{"depth": 1, "assertion": x['text'],
                           "proofs": re.split("\\s*\\bOR\\b\\s*", x['proofs'][2:-2])} for x in meta['allInferences']]
    else:
        # We exclude depth-0 "inferences", and otherwise keep order in terms of depth
        all_inferences = [x for x in from_all_proofs_field(meta['allProofs']) if x['depth'] != 0]

    if filter_naf_inferences:
        for inf in all_inferences:
            inf['proofs'] = [p for p in inf['proofs'] if "NAF" not in p]
        all_inferences = [x for x in all_inferences if len(x['proofs']) > 0]
    if len(all_inferences) == 0:
        all_inferences = [{"depth": 0, "assertion": NO_INFERENCE, "proofs": None}]
    proof_pns = []
    if proofs == "PN":
        for inf_proof in all_inferences:
            if inf_proof['proofs'] is None:
                proof_slot = [NO_PROOF]
            else:
                proof_strings = inf_proof['proofs']
                proof_strings.sort(key=lambda x: x.count("->"))
                proof_slot = [proof_to_polish_notation(p, token_map) for p in proof_strings]
                for (pn, p) in zip(proof_slot, proof_strings):
                    assert normalize_proof(polish_notation_to_proof(pn)) == normalize_proof(parse_lisp(p, token_map))
            proof_pns.append(proof_slot)
    slots_base = {"id": f"{theory_id}", 'context': context_string, 'question': question_text}
    meta_base = {"sentence_map": token_map}
    res = []
    if one_inference_per_answer and not inference_answer_as_list:
        for idx in range(len(all_inferences)):
            slots = slots_base.copy()
            slots["id"] = f"{theory_id}-{idx}"
            slots["answer"] = all_inferences[idx]['assertion']
            if proofs == "PN":
                slots['proof'] = proof_pns[idx]
            meta = meta_base.copy()
            meta['QDep'] = all_inferences[idx]['depth']
            slots['meta'] = meta
            res.append(slots)
    else:
        inferences = [x['assertion'] for x in all_inferences]
        depths = [x['depth'] for x in all_inferences]
        if inference_answer_as_list:
            slots_base["answer"] = inferences
        else:
            slots_base["answer"] = " ".join(inferences)
        meta_base['inferences'] = inferences
        if proofs == "PN":
            if inference_answer_as_list:
                slots_base["proof"] = [p[0] for p in proof_pns]
            else:
                slots_base["proof"] = " . ".join(p[0] for p in proof_pns)
            meta_base['all_proofs'] = proof_pns
        meta_base['QDep'] = depths
        slots_base['meta'] = meta_base
        res = [slots_base]
    return res


# Proofs in raw form from dataset, e.g. ((triple1 -> rule1) OR triple1)
def check_no_naf_in_proofs(proofs):
    for proof in proofs:
        if not 'NAF' in proof:
            return True
    return False

# Custom filtering function. category_fracs is e.g., {"depth-0": 0.3, "depth-1": 1.0}
def combine_ruletaker_meta_no_naf(ruletaker_dir, category_fracs, split, file_infix="",
                                  cull_frac_no_inference_wo_naf=0.0):
    res = []
    for category, frac in category_fracs.items():
        file = os.path.join(ruletaker_dir, category, f"meta-{file_infix}{split}.jsonl")
        if not os.path.exists(file):
            logger.warning(f"Cannot find file: {file}, skipping!")
            continue
        meta = load_jsonl(file)
        if frac < 1.0:
            meta = meta[:math.ceil(frac * len(meta))]
        if cull_frac_no_inference_wo_naf > 0:
            no_inf_ids = []
            for m in meta:
                all_inferences = [x for x in from_all_proofs_field(m['allProofs']) if x['depth'] != 0]
                found_non_naf_inference = False
                for inf_proof in all_inferences:
                    if check_no_naf_in_proofs(inf_proof['proofs']):
                        found_non_naf_inference = True
                        break
                if not found_non_naf_inference:
                    no_inf_ids.append(m['id'])
            num_cull = math.floor(cull_frac_no_inference_wo_naf * len(no_inf_ids))
            cull_ids = no_inf_ids[-num_cull:]
            old_meta = meta
            meta = []
            for m in old_meta:
                if m['id'] not in cull_ids:
                    meta.append(m)
        res += meta
    return res



def normalize_sentences(sentences, normalize_fn=None):
    if isinstance(sentences, str):
        # Streams of non-period characters starting with non-space.
        sentences = re.findall("[^. ][^.]+\\.", sentences)
    if normalize_fn is not None:
        sentences = [normalize_fn(s) for s in sentences]
    return sentences


# Score P/R/F1 for sentence overlaps, e.g., for RuleTaker inferences
def score_sentence_overlaps(sentences, sentences_gold, normalize_fn=None):
    sentences = normalize_sentences(sentences, normalize_fn)
    sentences_gold = normalize_sentences(sentences_gold, normalize_fn)
    print(f"\t\tsentences:{sentences}")
    print(f"\t\tsentences_gold:{sentences_gold}")
    if len(sentences) == 0 or len(sentences_gold) == 0:
        if sentences == sentences_gold:
            prec = recall = 1
        else:
            prec = recall = 0
    else:
        common = len(set(sentences).intersection(set(sentences_gold)))
        # Duplicates in sentences are penalized in precision
        prec = common / len(sentences)
        recall = common / len(sentences_gold)
    if prec == recall == 0:
        f1 = 0
    else:
        f1 = 2 * prec * recall / (prec + recall)
    acc = 0 if f1 < 1 else 1

    # if acc < 1.0:
    #     print(f"\n\n+++++++++++++++++++++++++")
    #     print(f"[score_sentence_overlaps]sentences:{sentences}")
    #     print(f"[score_sentence_overlaps]sentences_gold:{sentences_gold}")
    #     print(f"[score_sentence_overlaps]{acc}\t{f1}")
    return {"P": prec, "R": recall, "F1": f1, "acc": acc, "pred": sentences, "gold": sentences_gold}

def ruletaker_inferences_scores(slots, gold):
    from utils.eval_utils import squad_normalize_answer
    prediction = slots.get("answer")
    if prediction is None:
        return {"P": 0, "R": 0, "F1": 0, "acc": 0}
    if 'meta' in gold and 'inferences' in gold['meta']:
        sentences_gold = gold['meta']['inferences']
    else:
        sentences_gold = gold['answer']
    res = score_sentence_overlaps(prediction, sentences_gold, normalize_fn=squad_normalize_answer)
    return res


# Faux test suite:
proof1 = "((((((NAF) -> sent4) NAF sent5) -> sent7)) -> sent5)"
proof1_polish = proof_to_polish_notation(proof1)

assert proof1_polish == '# sent5 # sent7 & # sent4 NAF & NAF sent5'
assert normalize_proof(parse_lisp(proof1)) == normalize_proof(polish_notation_to_proof(proof1_polish))


fact_to_rep_patterns = []
for p in [
                "(a |an |the )?(?P<entity1>.+) (?P<reln>is|are) (a |an |the )?(?P<entity2>.+).",
                # "[^a|an|the]? (.+) (are) (.+).",
                "(a |an |the )?(?P<entity1>.+) (?P<reln>lives in) (a |an |the )?(?P<entity2>.+).",
                "(a |an |the )?(?P<entity1>.+) (?P<reln>.+?s) (a |an |the )?(?P<entity2>.+).",
            ]:
    fact_to_rep_patterns.append(re.compile(p, re.IGNORECASE))


# input: fact as text string
# output: fact in json format with representation filled in
def extract_fact_representation(fact):
    # Example output jsons
    # "triple6": {
    #       "text": "The dog chases the tiger.",
    #       "representation": "(\"dog\" \"chases\" \"tiger\" \"+\")"
    # }
    # "triple7": {
    #           "text": "The dog is nice.",
    #           "representation": "(\"dog\" \"is\" \"nice\" \"+\")"
    # }

    fact_json = {}
    # print(f"\n\n============\nfact:{fact}")
    if '#' in fact or '&' in fact:
        return fact_json
    for p in fact_to_rep_patterns:
        m = p.search(fact)
        if m:
            entity1 = m.group('entity1')
            reln = m.group('reln')
            entity2 = m.group('entity2')
            if entity1 and reln and entity2:
                repr_str = f"(\"{entity1}\" \"{reln}\" \"{entity2}\" \"+\")"
                fact_json = {
                    "text": fact,
                    "representation": repr_str
                }
                return fact_json
    return fact_json


# Get levels for each element in the proof
# e.g. input proof: P = [[[[['triple1', 'triple2'], '->', 'rule1'], 'triplem'], '->', 'rule2']]
# to get this format use parse_lisp for lisp_proof, or polish_notation_to_proof for polish proofs
# This function will return an object with levels when converted to list looks like this:
# [('triple1', 4), ('triple2', 4), ('->', 3), ('rule1', 3), ('triplem', 2), ('->', 1), ('rule2', 1)]
def levels(x: object, depth: object = -1) -> object:
    if not isinstance(x, list):
        yield (x, depth)
    else:
        for sublist in x:
            yield from levels(sublist, depth + 1)


def increment_count(depth_to_num_questions, key, increment):
    prev_count = depth_to_num_questions.get(key, 0.0)
    depth_to_num_questions[key] = prev_count + increment


def score_proof_polishPred_normalGold(pn_proof, gold_proofs):
    try:
        parsed_proof = polish_notation_to_proof(pn_proof)
    except:
        return {"acc": 0, "bad_parse": 1}
    try:
        gold_normalized = [normalize_proof(polish_notation_to_proof(proof_to_polish_notation(p))) for p in gold_proofs]
    except:
        raise ValueError(f"Bad gold proofs in {gold_proofs}")
    pred_normalized = normalize_proof(parsed_proof)
    score = 1 if pred_normalized in gold_normalized else 0
    #print(f"\n&&&&&&&&&&&&&&&&&&\n")
    #print(f"pred_normalized:{pred_normalized}")
    #print(f"gold_normalized:{gold_normalized}")
    #print(f"score:{score}")
    return {"acc": score}


def recursive_from_polish_notation_lenient(tokens, num_args=1, start_index=0):
    res = []
    i = start_index
    while i < len(tokens):
        symbol = tokens[i]
        if symbol in SYMBOL_DICT.values():
            args, end_index = recursive_from_polish_notation_lenient(tokens, num_args=2, start_index=i + 1)
            if symbol == SYMBOL_DICT['->']:
                res.append([args[1], "->", args[0]])
            elif symbol == SYMBOL_DICT['and']:
                res.append(args)
            i = end_index
        else:
            res.append(symbol)
            i += 1
        if len(res) == num_args:
            return res, i
    return res, i
    # raise ValueError("Not enough arguments found!")


def polish_notation_to_proof_lenient(polish_notation):
    tokens = polish_notation.split(" ")
    res, end_index = recursive_from_polish_notation_lenient(tokens)
    if end_index != len(tokens):
        return ""   # ERROR case: Full string was not processed!
    return res


def get_norm_mapped_proof_tokens(proof, token_map=None):
    token_regex = re.compile("(->|[a-zA-Z0-9%@^]+|\\S)")
    tokens = token_regex.findall(proof)
    if token_map is not None:
        tokens = [token_map.get(x, x) for x in tokens]
    return tokens


def get_set_of_sentences_from_proof(tokens):
    relevant_sentences = []
    ignore_tokens = ['(', ')', '->']
    for t in tokens:
        if t not in ignore_tokens and \
                        t not in relevant_sentences:
                relevant_sentences.append(t)

    return ', '.join(relevant_sentences)


def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / max(1, union)


def parse_entailment_step_proof_remove_ids(proof: str, slot_json_record: dict):
    sentences = []
    inferences = []
    int_to_all_ancestors = dict()
    int_to_all_ancestor_ids = dict()
    int_to_all_ancestors_list = []
    relevant_sentences = set()
    id_to_int = dict()
    id_to_sentence = slot_json_record['meta']['triples']
    print(f"PROOF:{proof}")

    if "[STEP]" in proof:
        temp = proof.split('[STEP] ', maxsplit=-1)
    else:
        temp = proof.split(';', maxsplit=-1)

    for t in temp:
        t_parts = t.strip().split(":")
        t = t_parts[0]
        int_str = ""
        if len(t_parts) == 2:
            int_str = t_parts[1].strip()

        if t:
            # normalize t by numerically sorting sentence ids in LHS
            t_parts = t.split(' -> ')
            if len(t_parts) == 2:
                rhs = t_parts[1].strip()
                if '&' in t_parts[0]:
                    lhs_ids = t_parts[0].split('&')
                else:
                    lhs_ids = t_parts[0].split(',')
                all_ancestors = set()
                all_ancestor_ids = set()
                lhs_ids = [lid.strip() for lid in lhs_ids]
                lhs_strs = [id_to_sentence.get(lid.strip(), "NULL") for lid in lhs_ids]
                #print(f"\t for rhs={rhs}")
                for lid in lhs_ids:
                    if 'sent' in lid:
                        # print(f"\t adding ancestor={lid}\tid_to_sentence:{id_to_sentence}")
                        l_sent = id_to_sentence.get(lid, 'NULL')
                        relevant_sentences.add(l_sent)
                        all_ancestor_ids.add(lid)
                        all_ancestors.add(l_sent)
                    else:
                        their_ancestor_ids = int_to_all_ancestor_ids.get(lid, set())
                        all_ancestor_ids = all_ancestor_ids.union(their_ancestor_ids)

                        their_ancestors = int_to_all_ancestors.get(lid, set())
                        all_ancestors = all_ancestors.union(their_ancestors)

                        #print(f"\t adding ancestors={their_ancestors}")

                # sorted_lhs_ids = sorted(lhs_ids)
                sorted_lhs_ids = sorted(lhs_strs)
                sorted_lhs = ' & '.join(sorted_lhs_ids)

                if rhs == "hypothesis":
                    int_str = slot_json_record['hypothesis']

                # sentences.append(f"{sorted_lhs} -> {rhs}")
                sentences.append(f"{sorted_lhs} -> {int_str}")
                #print(f"\t rhs = {rhs}, all_ancestors={all_ancestors}")


                id_to_int[rhs] = int_str
                id_to_sentence[rhs] = int_str
                int_to_all_ancestor_ids[rhs] = all_ancestor_ids
                int_to_all_ancestors[rhs] = all_ancestors

                int_to_all_ancestors_list.append(
                    {"int": rhs,
                    "ancestors":list(all_ancestors),
                    "ancestor_ids":list(all_ancestor_ids)
                     })
                inferences.append({
                    "lhs": sorted_lhs_ids,
                    "rhs": int_str
                })

    print(f"\n\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(f"proof:{proof}")
    print(f"sentences:{sentences}")
    print(f"inferences:{inferences}")
    print(f"int_to_all_ancestors_list:{int_to_all_ancestors_list}")
    print(f"relevant_sentences:{relevant_sentences}")
    print(f"id_to_int:{id_to_int}")
    return sentences, inferences, int_to_all_ancestors_list, relevant_sentences, id_to_int


def parse_entailment_step_proof(proof: str, gold_json_record: dict):
    sentences = []
    inferences = []
    int_to_all_ancestors = dict()
    int_to_all_ancestors_list = []
    relevant_sentences = set()
    id_to_int = dict()

    if "[STEP]" in proof:
        proof_steps = proof.split('[STEP] ', maxsplit=-1)
    else:
        proof_steps = proof.split(';', maxsplit=-1)

    for p_step in proof_steps:
        print(f"step:{p_step}")
        p_parts = p_step.strip().split(":")
        print(f"step_parts:{p_parts}")

        step = p_parts[0]
        int_str = ""
        if len(p_parts) == 2:
            from utils.entail_trees_utils import normalize_sentence
            int_str = normalize_sentence(p_parts[1].strip())

        if step:
            # normalize t by numerically sorting sentence ids in LHS
            step_parts = step.split(' -> ')
            if len(step_parts) == 2:
                rhs = step_parts[1]
                if '&' in step_parts[0]:
                    lhs_ids = step_parts[0].split('&')
                else:
                    lhs_ids = step_parts[0].split(',')
                all_ancestors = set()
                lhs_ids = [lid.strip() for lid in lhs_ids]
                #print(f"\t for rhs={rhs}")
                for lid in lhs_ids:
                    if 'sent' in lid:
                        relevant_sentences.add(lid)
                        all_ancestors.add(lid)
                        #print(f"\t adding ancestor={lid}")
                    else:
                        their_ancestors = int_to_all_ancestors.get(lid, set())
                        all_ancestors = all_ancestors.union(their_ancestors)
                        #print(f"\t adding ancestors={their_ancestors}")

                sorted_lhs_ids = sorted(lhs_ids)
                sorted_lhs = ' & '.join(sorted_lhs_ids)
                sentences.append(f"{sorted_lhs} -> {rhs}")
                print(f"lhs_ids:{lhs_ids}\t rhs = {rhs}\t all_ancestors={all_ancestors}")

                if rhs == "hypothesis":
                    from utils.entail_trees_utils import normalize_sentence
                    int_str = normalize_sentence(gold_json_record['hypothesis'])
                print(f"\t rhs = {rhs}, int_str={int_str}")

                id_to_int[rhs] = int_str
                int_to_all_ancestors[rhs] = all_ancestors
                int_to_all_ancestors_list.append(
                    {"int": rhs,
                    "ancestors":list(all_ancestors)
                     })
                inferences.append({
                    "lhs": sorted_lhs_ids,
                    "rhs": rhs
                })
    print(f"\t<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(f"\tproof:{proof}")
    print(f"\tsentences:{sentences}")
    print(f"\tinferences:{inferences}")
    print(f"\tint_to_all_ancestors_list:{int_to_all_ancestors_list}")
    print(f"\trelevant_sentences:{relevant_sentences}")
    print(f"\tid_to_int:{id_to_int}")
    return sentences, inferences, int_to_all_ancestors_list, relevant_sentences, id_to_int


def align_conclusions_across_proofs(int_to_all_ancestors_pred: list, int_to_all_ancestors_gold: list,
                                    id_to_int_pred: dict, id_to_int_gold: dict):
    pred_int_to_gold_int_mapping = dict()
    prediction_to_aligned_gold = dict()
    prediction_to_perfect_match = dict()

    # print(f"%%%%%%%%%\t[Inside align_conclusions_across_proofs]")
    # print(f"%%%%%%%%%\tint_to_all_ancestors_pred:{int_to_all_ancestors_pred}")
    # print(f"%%%%%%%%%\tint_to_all_ancestors_gold:{int_to_all_ancestors_gold}")
    # print(f"%%%%%%%%%\tid_to_int_pred:{id_to_int_pred}")
    # print(f"%%%%%%%%%\tid_to_int_gold:{id_to_int_gold}")

    for pred_int_json in int_to_all_ancestors_pred:
        pred_int = pred_int_json['int']
        pred_ancestors = pred_int_json['ancestors']
        prediction = id_to_int_pred[pred_int]
        # print(f"%%%%%%%%%\tpred_int_json:{pred_int_json}")
        # print(f"%%%%%%%%%\tpred_int:{pred_int}")
        # print(f"%%%%%%%%%\tpred_ancestors:{pred_ancestors}")

        max_sim = 0
        best_gold_int = ""
        for gold_int_json in int_to_all_ancestors_gold:
            gold_int = gold_int_json['int']
            gold_ancestors = gold_int_json['ancestors']

            jaccard_sim = jaccard_similarity(pred_ancestors, gold_ancestors)

            # print(f"%%%%%%%%%\tpred_int:{pred_int}")
            # print(f"%%%%%%%%%\tpred_ancestors:{pred_ancestors}")
            # print(f"%%%%%%%%%\tgold_int:{gold_int}")
            # print(f"%%%%%%%%%\tgold_ancestors:{gold_ancestors}")
            # print(f"%%%%%%%%%\tjaccard_sim:{jaccard_sim}")

            if jaccard_sim > max_sim:
                max_sim = jaccard_sim
                best_gold_int = gold_int

        if max_sim == math.isclose(max_sim, 1.0):
            prediction_to_perfect_match[prediction] = True
        else:
            prediction_to_perfect_match[prediction] = False

        if best_gold_int:
            pred_int_to_gold_int_mapping[pred_int] = best_gold_int
            prediction_to_aligned_gold[prediction] = id_to_int_gold[best_gold_int]
        else:
            pred_int_to_gold_int_mapping[pred_int] = "NO_MATCH"
            prediction_to_aligned_gold[prediction] = ""

    print(f"%%%%%%%%%\tpred_int_to_gold_int_mapping:{pred_int_to_gold_int_mapping}")
    return pred_int_to_gold_int_mapping, prediction_to_aligned_gold, prediction_to_perfect_match


def rewrite_aligned_proof(proof: str, pred_int_to_gold_int_mapping: dict):
    sentences_pred_aligned = []

    if "[STEP]" in proof:
        temp = proof.split('[STEP] ', maxsplit=-1)
    else:
        temp = proof.split('; ', maxsplit=-1)

    for t in temp:
        t = t.strip().split(":")[0]
        if t:
            # normalize t by numerically sorting sentence ids in LHS
            t_parts = t.split(' -> ')
            if len(t_parts) == 2:
                rhs_old = t_parts[1]
                rhs = pred_int_to_gold_int_mapping.get(rhs_old, rhs_old)
                if ' & ' in t_parts[0]:
                    lhs_ids = t_parts[0].split(' & ')
                else:
                    lhs_ids = t_parts[0].split(', ')
                aligned_lhs_ids = [pred_int_to_gold_int_mapping.get(lid, lid) for lid in lhs_ids]
                sorted_lhs_ids = sorted(aligned_lhs_ids)
                sorted_lhs = ' & '.join(sorted_lhs_ids)
                sentences_pred_aligned.append(f"{sorted_lhs} -> {rhs}")

    return sentences_pred_aligned


def rewrite_aligned_proof_noids(proof: str, pred_int_to_gold_int_mapping: dict,
                                pred_sentences: dict(), gold_ints: dict()
                                ):
    sentences_pred_aligned = []
    sentences_pred_aligned_strs = []
    print(f"pred_sentences:{pred_sentences}")
    print(f"gold_ints:{gold_ints}")
    pred_sentences.update(gold_ints)
    pred_sentences['NO_MATCH'] = 'NULL'

    if "[STEP]" in proof:
        temp = proof.split('[STEP] ', maxsplit=-1)
    else:
        temp = proof.split('; ', maxsplit=-1)

    for t in temp:
        t = t.strip().split(":")[0]
        if t:
            # normalize t by numerically sorting sentence ids in LHS
            t_parts = t.split(' -> ')
            if len(t_parts) == 2:
                rhs_old = t_parts[1]
                rhs = pred_int_to_gold_int_mapping.get(rhs_old, rhs_old).strip()
                rhs_str = pred_sentences[rhs]
                if ' & ' in t_parts[0]:
                    lhs_ids = t_parts[0].split(' & ')
                else:
                    lhs_ids = t_parts[0].split(', ')

                aligned_lhs_ids = [pred_int_to_gold_int_mapping.get(lid.strip(), lid.strip()) for lid in lhs_ids]
                aligned_lhs_strs = [pred_sentences[pred_int_to_gold_int_mapping.get(lid.strip(), lid.strip())] for lid in lhs_ids]

                sorted_lhs_ids = sorted(aligned_lhs_ids)
                sorted_lhs = ' & '.join(sorted_lhs_ids)
                sorted_lhs_strs = ' & '.join(sorted(aligned_lhs_strs))
                sentences_pred_aligned.append(f"{sorted_lhs} -> {rhs}")
                sentences_pred_aligned_strs.append(f"{sorted_lhs_strs} -> {rhs_str}")

    return sentences_pred_aligned, sentences_pred_aligned_strs
