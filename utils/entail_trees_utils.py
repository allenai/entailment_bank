from collections import defaultdict
from copy import deepcopy
import random
import re

from utils.proof_utils import parse_lisp, decouple_proof_struct_ints, polish_notation_to_proof


# Count phrase appearing in a reference string, making sure word boundaries are respected
def count_phrase_matches(phrase, reference):
    regex = "(\\b|(?!\\w))" + re.escape(phrase) + "((?<!\\w)|\\b)"
    return len(re.findall(regex, reference))


def sentence_index(sentence_id):
    return int(sentence_id.replace('sent', ''))


def sort_sentence_ids_match(match):
    sentence_ids = match.group(0).split(" & ")
    sentence_ids.sort(key=sentence_index)
    return " & ".join(sentence_ids)


# Scramble sentences within a entail_tree dataset entry
def scramble_sentences_in_entail_tree_q(qdata):
    sentences = qdata['meta']['triples']
    sentences_shuffled = list(sentences.keys())
    random.shuffle(sentences_shuffled)
    sentence_map = dict(zip(sentences.keys(), sentences_shuffled))
    return remap_sentences(qdata, sentence_map)


def remap_sentences(qdata, sentence_map):
    old_proof = qdata['proof']
    new_proof = re.sub('sent\\d+', lambda x:sentence_map[x.group(0)], old_proof)
    new_proof = re.sub('(sent\\d+ & )+sent\\d+', sort_sentence_ids_match, new_proof)
    sentences = qdata['meta']['triples']
    new_sentences = [(sentence_map[k], v) for k,v in sentences.items()]
    new_sentences.sort(key=lambda x: sentence_index(x[0]))
    new_sentences = dict(new_sentences)
    res = qdata.copy()
    res['context'] = " ".join([f"{k}: {v}" for k,v in new_sentences.items()])
    res['proof'] = new_proof
    res['meta']['triples'] = new_sentences
    if 'distractors' in res['meta']:
        new_distractors = [sentence_map[x] for x in res['meta']['distractors']]
        #new_distractors.sort(key=sentence_index)
        res['meta']['distractors'] = new_distractors
    return res


def get_parents_recursive(proof, dependencies):
    if isinstance(proof, str):
        return dependencies
    if len(proof) == 3 and proof[1] == "->":
        new_dependency = proof[2]
        for k, v in dependencies.items():
            dependencies[k] = v + [new_dependency]
        if new_dependency not in dependencies:
            dependencies[new_dependency] = []
        return get_parents_recursive(proof[0], dependencies)
    elif len(proof) == 0:
        return dependencies
    elif len(proof) == 1:
        return get_parents_recursive(proof[0], dependencies)
    else:
        all_dependencies = []
        for sub_proof in proof:
            dep1 = deepcopy(dependencies)
            get_parents_recursive(sub_proof, dep1)
            all_dependencies.append(dep1)
        dd = defaultdict(list)
        for d in all_dependencies:
            for key, value in d.items():
                dd[key] += value
        return dd


def get_intermediate_dependencies(proof):
    list_proof = parse_lisp(proof)
    dependencies = {}
    res = get_parents_recursive(list_proof, dependencies)
    return res


def get_stripped_recursive(proof, stripped):
    if isinstance(proof, str):
        return proof
    if len(proof) == 3 and proof[1] == "->":
        stripped[proof[2]] = [get_stripped_recursive(x, stripped) for x in proof[0]]
        return proof[2]
    elif len(proof) == 1:
        return get_stripped_recursive(proof[0], stripped)
    else:
        raise ValueError(f"Nonsense found: {proof}")


def get_core_proofs(proof):
    list_proof = parse_lisp(proof)
    stripped = {}
    get_stripped_recursive(list_proof, stripped)
    return stripped


def remove_distractors(qdata, num_removed_distractors=0):
    if num_removed_distractors < 1:
        return qdata
    else:
        new_q = deepcopy(qdata)
        distractors = new_q['meta']['distractors']
        sentences_removed = list(reversed(distractors))[:num_removed_distractors]
        sentences_remaining = {k: v for k, v in new_q['meta']['triples'].items() if k not in sentences_removed}
        new_distractors = [k for k in new_q['meta']['distractors'] if k not in sentences_removed]
        sentence_map = {k: f"sent{i + 1}" for i, k in enumerate(sentences_remaining.keys())}
        new_q['meta']['triples'] = sentences_remaining
        new_q['meta']['distractors'] = new_distractors
        new_q = remap_sentences(new_q, sentence_map)
        return new_q


# Break down an entailment tree data instance into one-step inference steps
def make_inference_steps(qdata, rescramble_sentences=False, num_removed_distractors=0):
    proof = qdata['meta']['lisp_proof']
    core_proofs = list(get_core_proofs(proof).items())
    random.shuffle(core_proofs)
    sentences = qdata['meta']['triples'].copy()
    intermediates = qdata['meta']['intermediate_conclusions']
    q_id = qdata['id']
    hypothesis_id = qdata['meta']['hypothesis_id']
    res = []
    while len(core_proofs) > 0:
        selected = None
        for proof in core_proofs:
            selected = proof
            for dep in proof[1]:
                if 'int' in dep:
                    selected = None
            if selected is not None:
                break
        if selected is None:
            raise ValueError(f"No resolved proofs in {core_proofs}")
        new_res = selected[0]
        if new_res == hypothesis_id:
            new_res_text = "hypothesis"
            assert len(core_proofs) == 1
        else:
            new_res_text = "int1: " + intermediates[new_res]

        new_proof = selected[1]
        new_proof_text = " & ".join(new_proof) + " -> " + new_res_text
        new_context = " ".join([f"{k}: {v}" for k, v in sentences.items()])
        new_q = deepcopy(qdata)
        new_q['id'] = f"{q_id}-add{len(res)}"
        new_q['meta'] = {'triples': sentences.copy(),
                         'distractors': new_q['meta'].get('distractors', [])}
        new_q['proof'] = new_proof_text
        new_q['meta']['hypothesis_id'] = "int1"
        new_q['depth_of_proof'] = 1
        new_q['length_of_proof'] = 1
        new_q['context'] = new_context
        if rescramble_sentences:
            new_q = scramble_sentences_in_entail_tree_q(new_q)
        if num_removed_distractors > 0:
            new_q = remove_distractors(new_q, num_removed_distractors)

        res.append(new_q)
        new_sentence = "sent" + str(len(sentences) + 1)
        sentences[new_sentence] = intermediates[selected[0]]
        new_core_proofs = []
        for proof in core_proofs:
            if proof[0] == new_res:
                continue
            new_parents = []
            for parent in proof[1]:
                new_parents.append(new_sentence if parent == new_res else parent)
            new_core_proofs.append((proof[0], new_parents))
        core_proofs = new_core_proofs
    return res


def normalize_sentence(sent):
    return sent.replace("  ", " ").replace(".", "").replace('\n', '').replace("( ", "").replace(" )", "").lower().strip()


def get_entailment_steps_from_polish_proof(polish_proof):
    print(f"POLISH_PROOF:{polish_proof}")
    pn_without_ints, int_dict = decouple_proof_struct_ints(polish_proof)
    print(f"POLISH_PROOF without INTS:{pn_without_ints}")
    try:
        recursive_proof = polish_notation_to_proof(pn_without_ints)[0]
    except:
        return []
    print(f"recursive_proof:{recursive_proof}")
    return get_entailment_steps_from_recursive_proof(recursive_proof, int_dict)


def append_list(list_obj, to_be_added):
    if isinstance(to_be_added, list):
        for add_item in to_be_added:
            if add_item != '->':
                print(f"\t**********adding {add_item} to lhs")
                list_obj += append_list(list_obj, add_item)
    else:
        if to_be_added != '->':
            print(f"\t**********adding {to_be_added} to lhs")
            list_obj.append(to_be_added)
    return list_obj


def get_entailment_steps_from_recursive_proof(recursive_proof, int_dict):
    entailment_steps = []
    print(f"======Calling recursion: recursive_proof:{recursive_proof}")

    lhs = recursive_proof[0]
    rhs = recursive_proof[2]
    rhs_str = int_dict.get(rhs, "")
    print(f"======lhs:{lhs}")
    print(f"======rhs:{rhs}")
    lhs_ids = []
    if isinstance(lhs, str):
        append_list(lhs_ids, lhs)
    else:
        for l in lhs:
            print(f"\tl:{l}")
            if isinstance(l, str):
                append_list(lhs_ids, l)
            else:
                if '->' in l:
                    entailment_steps += get_entailment_steps_from_recursive_proof(l, int_dict)
                    append_list(lhs_ids, l[2])
                else:
                    print(f"\t^^^^lhs:{lhs}")
                    print(f"\t^^^^rhs:{rhs}")
                    print(f"\t^^^^l{l}")
                    print(f"\t^^^^^lhs_ids{lhs_ids}")
                    for l_part in l:
                        if isinstance(l_part, list):
                            if '->' in l_part:
                                entailment_steps += get_entailment_steps_from_recursive_proof(l_part, int_dict)
                                append_list(lhs_ids, l_part[2])
                            else:
                                append_list(lhs_ids, l_part)
                        else:
                            append_list(lhs_ids, l_part) # for cases like ['sent20', 'sent8'] in [['sent14', ['sent20', 'sent8']], '->', 'int1']
                    print(f"\tlhs_ids{lhs_ids}")

    print(f"\tlhs_ids:{lhs_ids}")
    print(f"\trhs:{rhs}")
    lhs_str = ' & '.join(lhs_ids)
    print(f"++++Adding step: {lhs_str} -> {rhs}: {rhs_str}")
    entailment_steps.append(f"{lhs_str} -> {rhs}: {rhs_str}")
    return entailment_steps
