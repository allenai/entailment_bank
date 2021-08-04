from collections import defaultdict
from copy import deepcopy
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import random
import re

from utils.proof_utils import parse_lisp, decouple_proof_struct_ints, polish_notation_to_proof

porter_stemmer = PorterStemmer()

# Count phrase appearing in a reference string, making sure word boundaries are respected
def count_phrase_matches(phrase, reference):
    regex = "(\\b|(?!\\w))" + re.escape(phrase) + "((?<!\\w)|\\b)"
    return len(re.findall(regex, reference))


# Use NLTK to stem each word
def stem_all_words(text):
    return " ".join(porter_stemmer.stem(w) for w in word_tokenize(text))


# Heuristics for counting phrases in a list of references, putting more emphasis on
# the earlier references, and first checking exact match before stemmed match
def best_phrase_match(phrases, references):
    if len(phrases) == 1:
        return phrases[0]
    scored_phrases = []
    for phrase in phrases:
        multiplier = 1
        score = 0
        for reference in references:
            score += multiplier * count_phrase_matches(phrase, reference)
            multiplier *= 0.9  # discount a bit later references
        scored_phrases.append((phrase, score))
    scored_phrases.sort(key=lambda x: -x[1])
    # Check if we have found a match without any ties
    if scored_phrases[0][1] > 0 and scored_phrases[1][1] < scored_phrases[0][1]:
        return scored_phrases[0][0]
    # If still a tie, we look for stemmed phrases
    stemmed_references = [stem_all_words(x) for x in references]
    scored_phrases_stem = []
    for (phrase, score) in scored_phrases:
        multiplier = 1
        for reference in stemmed_references:
            score += multiplier * count_phrase_matches(stem_all_words(phrase), reference)
            multiplier *= 0.9  # discount a bit later references
        scored_phrases_stem.append((phrase, score))
    scored_phrases_stem.sort(key=lambda x: -x[1])
    # Check if we have found a match without any ties
    if scored_phrases_stem[0][1] > 0 and scored_phrases_stem[1][1] < scored_phrases_stem[0][1]:
        return scored_phrases_stem[0][0]
    # If still tied, we count individual stemmed word matches rather than phrases
    scored_phrases_stem_words = []
    for (phrase, score) in scored_phrases_stem:
        multiplier = 1
        for reference in stemmed_references:
            for word in word_tokenize(stem_all_words(phrase)):
                score += multiplier * count_phrase_matches(word, reference)
            multiplier *= 0.9  # discount a bit later references
        scored_phrases_stem_words.append((phrase, score))
    scored_phrases_stem_words.sort(key=lambda x: -x[1])
    if scored_phrases_stem_words[0][1] > 0 and scored_phrases_stem_words[1][1] < scored_phrases_stem_words[0][1]:
        return scored_phrases_stem_words[0][0]
    # If still tied, just default to the earliest phrases with the highest score
    max_score = scored_phrases_stem_words[0][1]
    scored_phrases_stem_words_dict = dict(scored_phrases_stem_words)
    for phrase in phrases:
        if scored_phrases_stem_words_dict[phrase] == max_score:
            return phrase
    return phrase[0]  # should never get here


def replace_phrase_pattern(match, references):
    phrases = match.group(0)[1:-1].split(" / ")
    phrases = [x.strip() for x in phrases]
    return best_phrase_match(phrases, references)


# Resolve alternate phrases in question sentences for the entail_tree dataset structure
def resolve_sentence_alts(qdata, update_core_concepts=True):
    sentences = qdata['meta']['triples']
    ints = qdata['meta']['intermediate_conclusions']
    ints_reversed = list(reversed(list(ints.values())))
    proof = qdata['proof']
    alt_regex = "\\( [^\\)/]+ / [^\\)]+ \\)"
    ints_reversed_unambig = [x for x in ints_reversed if not re.findall(alt_regex, x)]
    if len(ints_reversed_unambig) < len(ints_reversed):
        # If any intermediates have ambiguities, resolve those first
        references = ints_reversed_unambig + [qdata['answer'], qdata['question']]
        new_ints = {}
        for int_id, int_val in ints.items():
            new_int = re.sub(alt_regex, lambda x: replace_phrase_pattern(x, references), int_val)
            if new_int != int_val:
                proof = proof.replace(f"{int_id}: {int_val}",f"{int_id}: {new_int}")
            new_ints[int_id] = new_int
        ints = new_ints
        ints_reversed = list(reversed(list(ints.values())))

    references = ints_reversed + [qdata['answer'], qdata['question']]
    new_sentences = {}
    for sentence_id, sentence in sentences.items():
        new_sentence = re.sub(alt_regex, lambda x: replace_phrase_pattern(x, references), sentence)
        new_sentences[sentence_id] = new_sentence
    res = deepcopy(qdata)
    if update_core_concepts:
        new_core_concepts = []
        for sentence in qdata['meta']['core_concepts']:
            new_core_concepts.append(
                re.sub(alt_regex, lambda x: replace_phrase_pattern(x, references), sentence))
        res['meta']['core_concepts'] = new_core_concepts
    res['meta']['triples'] = new_sentences
    res['meta']['intermediate_conclusions'] = ints
    new_context = " ".join(f"{k}: {v}" for k,v in new_sentences.items())
    res['context'] = new_context
    proof = proof.strip()  #remove space at end
    res['meta']['step_proof'] = proof
    res['proof'] = proof
    return res


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



######################
## example tests

example_q = {'id': 'NYSEDREGENTS_2014_4_2',
 'context': 'sent1: earth is a kind of planet. sent2: the sun is a kind of star. sent3: the earth revolves around the sun. ' + \
            'sent4: a complete ( revolution / orbit ) of a planet around its star takes ( 1 / one ) planetary year.',
 'question': 'About how long does it take Earth to make one revolution around the Sun?',
 'answer': 'a year',
 'hypothesis': 'a complete revolution of the earth around the sun will take one earth year',
 'proof': 'sent1 & sent2 & sent3 -> int1: the earth revolving around the sun is an example of a planet revolving around its star; int1 & sent4 -> hypothesis; ',
 'depth_of_proof': 2,
 'length_of_proof': 2,
 'meta': {'question_text': 'About how long does it take Earth to make one revolution around the Sun?',
  'answer_text': 'a year',
  'hypothesis_id': 'int2',
  'triples': {'sent1': 'earth is a kind of planet.',
   'sent2': 'the earth revolves around the sun.',
   'sent3': 'the sun is a kind of star.',
   'sent4': 'a complete ( revolution / orbit ) of a planet around its star takes ( 1 / one ) planetary year.'},
  'intermediate_conclusions': {'int1': 'the earth revolving around the sun is an example of a planet revolving around its star',
   'int2': 'a complete revolution of the earth around the sun will take one earth year'},
  'core_concepts': ['a complete ( revolution / orbit ) of a planet around its star takes ( 1 / one ) planetary year'],
  'step_proof': 'sent1 & sent2 & sent3 -> int1: the earth revolving around the sun is an example of a planet revolving around its star; int1 & sent4 -> hypothesis; ',
  'lisp_proof': '((((sent1 sent2 sent3) -> int1) sent4) -> int2)',
  'polish_proof': '# int2 & # int1 & sent1 & sent2 sent3 sent4'}}

resolved_q = resolve_sentence_alts(example_q)

assert resolved_q['context'] == 'sent1: earth is a kind of planet. sent2: the earth revolves around the sun. ' + \
  'sent3: the sun is a kind of star. sent4: a complete revolution of a planet around its star takes one planetary year.'


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
