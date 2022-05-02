import collections
import copy
import json
import re
import rouge
import string

from utils.angle_utils import decompose_slots
from utils.proof_utils import parse_entailment_step_proof, \
    align_conclusions_across_proofs, rewrite_aligned_proof, score_sentence_overlaps, ruletaker_inferences_scores, \
    parse_entailment_step_proof_remove_ids, rewrite_aligned_proof_noids

INCLUDE_NLG_EVAL = False

if INCLUDE_NLG_EVAL:
    from nlgeval import NLGEval
    # Initialize:
    nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)

def load_slot_data_by_id(slot_file):
    res = {}
    with open(slot_file) as file:
        for line in file:
            data = json.loads(line.strip())
            res[data['id']] = data
    return res

USE_GOOGLE_ROUGE_CODE = False

if USE_GOOGLE_ROUGE_CODE:
#     The dependencies in https://github.com/google-research/google-research/blob/master/rouge/requirements.txt
    from rouge_score import rouge_scorer
    rouge_scoring_fun = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def rouge_google_metric_max_over_ground_truths(prediction, ground_truths):
        if len(ground_truths) == 0:
            return 0
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = rouge_scoring_fun.score(prediction, ground_truth)
            scores_for_ground_truths.append(score['rougeL'].fmeasure)
        return max(scores_for_ground_truths)


def score_string_similarity(str1, str2):
    if str1 == str2:
        return 3.0   # Better than perfect token match
    str1 = fix_t5_unk_characters(str1)
    str2 = fix_t5_unk_characters(str2)
    if str1 == str2:
        return 2.0
    str1 = str1.lower()
    str2 = str2.lower()
    if str1 == str2:
        return 1.5
    if " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        return 0.0


def replace_punctuation(str):
    return str.replace("\"", "").replace("'", "")

# remove characters tokenized as unknown (\u2047) character
T5_GOOD_CHARS=[32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66,
    67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
    84, 85, 86, 87, 88, 89, 90, 91, 93, 95, 97, 98, 99, 100, 101, 102,
    103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
    117, 118, 119, 120, 121, 122, 124, 163, 171, 173, 174, 176, 187, 201,
    206, 220, 223, 224, 225, 226, 228, 231, 232, 233, 234, 238, 243, 244,
    246, 249, 251, 252, 259, 351, 355, 537, 539, 1072, 1074, 1076, 1077,
    1080, 1082, 1083, 1084, 1085, 1086, 1088, 1089, 1090, 1091, 8211,
    8212, 8216, 8217, 8220, 8221, 8222, 8226, 8242, 8364, 9601]
T5_BAD_REGEX = re.compile("[^"+re.escape(".".join([chr(x) for x in T5_GOOD_CHARS]))+"]")


def fix_t5_unk_characters(str):
    return re.sub(" {2,}", " ", re.sub(T5_BAD_REGEX, " ", str))

# Rouge evaluator copied from UnifiedQA
rouge_l_evaluator = rouge.Rouge(
    metrics=["rouge-l"],
    max_n=4,
    limit_length=True,
    length_limit=100,
    length_limit_type="words",
    apply_avg=True,
    apply_best=True,
    alpha=0.5,
    weight_factor=1.2,
    stemming=True,
)

def rouge_l(p, g):
    return rouge_l_evaluator.get_scores(p, g)

def rouge_metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, [ground_truth])
        scores_for_ground_truths.append(score)
    if isinstance(score, dict) and "rouge-l" in score:
        max_score = copy.deepcopy(score)
        max_score["rouge-l"]["f"] = round(
            max([score["rouge-l"]["f"] for score in scores_for_ground_truths]), 2
        )
        max_score["rouge-l"]["p"] = round(
            max([score["rouge-l"]["p"] for score in scores_for_ground_truths]), 2
        )
        max_score["rouge-l"]["r"] = round(
            max([score["rouge-l"]["r"] for score in scores_for_ground_truths]), 2
        )
        return max_score
    else:
        return round(max(scores_for_ground_truths), 2)

def nlg_string_similarities(prediction, gold_strings, normalize=True):
    if gold_strings is None:
        res = {"missing_gold": 1}
        if prediction is None:
            res['missing'] = 1
        return res
    if prediction is None:
        return {"missing": 1}
    if normalize:
        gold_strings = [fix_t5_unk_characters(x.lower()) for x in gold_strings]
        prediction = fix_t5_unk_characters(prediction.lower())
    # gold_strings = gold_strings[:1]
    res = {}
    if INCLUDE_NLG_EVAL:
        res = nlgeval.compute_individual_metrics(gold_strings, prediction)
        if 'CIDEr' in res:
            del res['CIDEr']
    rouge_l_score = rouge_metric_max_over_ground_truths(rouge_l, prediction, gold_strings)
    res['ROUGE_L_F'] = rouge_l_score["rouge-l"]["f"]
    if USE_GOOGLE_ROUGE_CODE:
        res['ROUGE_L_G'] = rouge_google_metric_max_over_ground_truths(prediction, gold_strings)
    res['pred'] = prediction
    res['gold'] = gold_strings
    if not gold_strings[0] and not prediction:
        res['ROUGE_L_F'] = 1.0
    return res


def nlg_string_similarities_intermediates_with_F1(prediction_to_aligned_gold: dict,
                                          id_to_int_gold: dict(),
                                          id_to_int_pred: dict(),
                                          prediction_to_perfect_match: dict,
                                          normalize=True,
                                          bleurt_scorer=None,
                                          bleurt_threshold=0.28 #for original BLEURT
                                        ):
    num_perfect_aligns = 0

    sum_rouge_l_score = 0.0
    sum_perfect_align_rouge_l_score = 0.0
    sum_bleurt_score = 0.0
    sum_perfect_align_bleurt_score = 0.0
    num_bleurt_correct = 0.0
    num_perfect_align_bleurt_correct = 0.0

    preds = []
    golds = []
    res = {}
    # print(f"prediction_to_aligned_gold:{prediction_to_aligned_gold}")
    pred_precise = set()
    gold_covered = set()
    for prediction, gold in prediction_to_aligned_gold.items():
        preds.append(prediction)
        golds.append(gold)
        gold_strings = [gold]
        if normalize:
            gold_strings = [fix_t5_unk_characters(x.lower()) for x in gold_strings]
            prediction = fix_t5_unk_characters(prediction.lower())
        #res = nlgeval.compute_individual_metrics(gold_strings, prediction)
        #if 'CIDEr' in res:
        #    del res['CIDEr']
        rouge_l_score = rouge_metric_max_over_ground_truths(rouge_l, prediction, gold_strings)
        if bleurt_scorer:
            bleurt_score = bleurt_scorer.score(references=gold_strings, candidates=[prediction], batch_size=1)[0]
        else:
            bleurt_score = -1
        # bleurt_score = max(0.0, min(1.0, unnorm_bleurt_score))
        # bleurt_score = [0.0]
        sum_rouge_l_score += rouge_l_score["rouge-l"]["f"]
        sum_bleurt_score += bleurt_score
        # if gold == "":
        #     print(f"@@@@@@@@@@@@@@@@@@@@@@@PREDICTION:{prediction}\tGOLD:{gold}\tBLEURT:{bleurt_score}")
        # print(f"bleurt_score:{unnorm_bleurt_score}\t{bleurt_score}")

        if bleurt_score >= bleurt_threshold:
            if gold != "":
                num_bleurt_correct += 1
                pred_precise.add(prediction)
                gold_covered.add(gold)

        # print(f"prediction_to_perfect_match:{prediction_to_perfect_match}")
        # print(f"prediction:{prediction}")
        if prediction_to_perfect_match[prediction]:
            sum_perfect_align_rouge_l_score += rouge_l_score["rouge-l"]["f"]
            sum_perfect_align_bleurt_score += bleurt_score
            num_perfect_aligns += 1
            if bleurt_score >= bleurt_threshold:
                num_perfect_align_bleurt_correct += 1

    bleurt_P = len(pred_precise)/max(1, len(id_to_int_pred))
    bleurt_R = len(gold_covered)/max(1, len(id_to_int_gold))
    if (bleurt_P + bleurt_R) == 0.0:
        bleurt_F1 = 0.0
    else:
        bleurt_F1 = (2 * bleurt_P * bleurt_R) / (bleurt_P + bleurt_R)
    # print(f"@@@@@@@@@@@@@@@@@@@@@@@pred_precise:{len(pred_precise)}\tpred:{len(id_to_int_pred)}\tgold_covered:{len(gold_covered)}\tgold:{len(id_to_int_gold)}")
    # print(f"@@@@@@@@@@@@@@@@@@@@@@@bleurt_P:{bleurt_P}\tbleurt_R:{bleurt_R}\tbleurt_F1:{bleurt_F1}")
    res['ROUGE_L_F'] = sum_rouge_l_score / max(1, len(prediction_to_aligned_gold.keys()))
    res['ROUGE_L_F_perfect_align'] = sum_perfect_align_rouge_l_score / max(1, num_perfect_aligns)
    res['BLEURT'] = sum_bleurt_score / max(1, len(prediction_to_aligned_gold.keys()))
    res['BLEURT_perfect_align'] = sum_perfect_align_bleurt_score / max(1, num_perfect_aligns)
    res['BLEURT_P'] = bleurt_P
    res['BLEURT_R'] = bleurt_R
    res['BLEURT_F1'] = bleurt_F1
    # res['BLEURT_acc'] = int(num_bleurt_correct == len(prediction_to_aligned_gold.keys()))
    res['BLEURT_acc'] = int(bleurt_F1==1)
    res['BLEURT_acc_perfect_align'] = int(num_perfect_align_bleurt_correct == num_perfect_aligns)
    res['fraction_perfect_align'] = num_perfect_aligns/max(1, len(prediction_to_aligned_gold.keys()))
    res['pred'] = preds
    res['gold'] = golds
    # print(f"res:{res}")
    return res


def nlg_string_similarities_intermediates(prediction_to_aligned_gold: dict,
                                          prediction_to_perfect_match: dict,
                                          normalize=True,
                                          bleurt_scorer=None,
                                          bleurt_threshold=0.28 #for original BLEURT
                                        ):
    num_perfect_aligns = 0

    sum_rouge_l_score = 0.0
    sum_perfect_align_rouge_l_score = 0.0
    sum_bleurt_score = 0.0
    sum_perfect_align_bleurt_score = 0.0
    num_bleurt_correct = 0.0
    num_perfect_align_bleurt_correct = 0.0

    preds = []
    golds = []
    res = {}
    # print(f"prediction_to_aligned_gold:{prediction_to_aligned_gold}")
    for prediction, gold in prediction_to_aligned_gold.items():
        preds.append(prediction)
        golds.append(gold)
        gold_strings = [gold]
        if normalize:
            gold_strings = [fix_t5_unk_characters(x.lower()) for x in gold_strings]
            prediction = fix_t5_unk_characters(prediction.lower())
        #res = nlgeval.compute_individual_metrics(gold_strings, prediction)
        #if 'CIDEr' in res:
        #    del res['CIDEr']
        rouge_l_score = rouge_metric_max_over_ground_truths(rouge_l, prediction, gold_strings)
        if bleurt_scorer:
            bleurt_score = bleurt_scorer.score(gold_strings, [prediction], batch_size=1)[0]
        else:
            bleurt_score = -1
        # bleurt_score = max(0.0, min(1.0, unnorm_bleurt_score))
        # bleurt_score = [0.0]
        sum_rouge_l_score += rouge_l_score["rouge-l"]["f"]
        sum_bleurt_score += bleurt_score
        # print(f"bleurt_score:{unnorm_bleurt_score}\t{bleurt_score}")

        if bleurt_score >= bleurt_threshold:
            num_bleurt_correct += 1
        # print(f"prediction_to_perfect_match:{prediction_to_perfect_match}")
        # print(f"prediction:{prediction}")
        if prediction_to_perfect_match[prediction]:
            sum_perfect_align_rouge_l_score += rouge_l_score["rouge-l"]["f"]
            sum_perfect_align_bleurt_score += bleurt_score
            num_perfect_aligns += 1
            if bleurt_score >= bleurt_threshold:
                num_perfect_align_bleurt_correct += 1

    res['ROUGE_L_F'] = sum_rouge_l_score / max(1, len(prediction_to_aligned_gold.keys()))
    res['ROUGE_L_F_perfect_align'] = sum_perfect_align_rouge_l_score / max(1, num_perfect_aligns)
    res['BLEURT'] = sum_bleurt_score / max(1, len(prediction_to_aligned_gold.keys()))
    res['BLEURT_perfect_align'] = sum_perfect_align_bleurt_score / max(1, num_perfect_aligns)
    res['BLEURT_acc'] = int(num_bleurt_correct == len(prediction_to_aligned_gold.keys()))
    res['BLEURT_acc_perfect_align'] = int(num_perfect_align_bleurt_correct == num_perfect_aligns)
    res['fraction_perfect_align'] = num_perfect_aligns/max(1, len(prediction_to_aligned_gold.keys()))
    res['pred'] = preds
    res['gold'] = golds
    # print(f"res:{res}")
    return res


def squad_normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return fix_t5_unk_characters(white_space_fix(remove_articles(remove_punc(lower(s)))))

def get_tokens(s):
    if not s: return []
    return squad_normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(squad_normalize_answer(a_gold) == squad_normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def split_mcoptions(mcoptions):
    first_option = ord(mcoptions.strip()[1])
    labels = "".join([chr(x) for x in range(first_option, first_option+10)])
    choices = re.split("\\s*\\(["+labels+"]\\)\\s*", mcoptions)[1:]
    return (choices, chr(first_option))


def mcq_answer_accuracy(slots, gold):
    answer = slots.get('answer')
    if answer is None:
        return {"acc": 0, "missing": 1}
    mcoptions, first_label = split_mcoptions(gold['mcoptions'])
    best = -1
    selected = None
    selected_key = None
    for idx, option in enumerate(mcoptions):
        score = score_string_similarity(answer, option)
        if score > best:
            best = score
            selected = option
            selected_key = chr(ord(first_label) + idx)
    acc = 1 if selected == gold['answer'] else 0
    res = {"acc": acc, "answerkey": selected_key, "align_score": best}
    return res


def bool_accuracy(slots, gold):
    pred_answer = str(slots.get('answer'))
    gold_answer = str(gold['answer'])

    res = {"acc": float(pred_answer==gold_answer),
           "ROUGE_L_F": float(pred_answer==gold_answer)}
    #print(f"---- {type(pred_answer)}\t{type(gold_answer)}\t{res}")
    return res


def squad_em_f1(answer, gold_answers):
    best_em = -1
    best_f1 = -1
    best_match = ""
    for gold in gold_answers:
        if gold.lower() == "noanswer":
            if answer.strip().lower() == "noanswer":
                em = f1 = 1.0
            else:
                em = f1 = 0.0
        else:
            em = compute_exact(gold, answer)
            f1 = compute_f1(gold, answer)
        if em > best_em:
            best_em = em
            best_match = gold
        if f1 > best_f1:
            best_f1 = f1
            best_match = gold
    res = {"EM": best_em, "F1": best_f1, "matched_gold": best_match}
    return res


def rc_answer_accuracy(slots, gold, slot_name='answer'):
    answer = str(slots.get(slot_name))
    if answer is None:
        return {"EM": 0, "F1": 0, "missing": 1}
    gold_answers = str(gold[slot_name])
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]
    return squad_em_f1(answer, gold_answers)


def extact_string_match_accuracy(slots, gold, slot_name='answer'):
    answer = str(slots.get(slot_name))
    if answer is None:
        return {"EM": 0, "F1": 0, "missing": 1}
    gold_answers = str(gold[slot_name])
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]
    return exact_match(answer, gold_answers)


def exact_match(answer, gold_answers):
    best_em = -1
    best_match = ""
    for gold in gold_answers:
        em = 1.0 if gold.lower() == answer.lower() else 0.0
        if em > best_em:
            best_em = em
            best_match = gold

    res = {"EM": best_em, "matched_gold": best_match}
    return res


def basic_split_mcoptions(mcoptions):
    splits = re.split("\\s*\\(\\w\\)\\s*", mcoptions)
    res = [s for s in splits if s != '']
    return res


# Very basic matching algorithm to gold MC options, not very meaningful
# (finds best matching gold option, only best score kept for each matched olgd)
def rough_mcoptions_f1(pred_mcoptions, gold_mcoptions):
    if pred_mcoptions is None:
        return {"F1": 0, "missing": 1}
    gold_split = basic_split_mcoptions(gold_mcoptions)
    pred_split = basic_split_mcoptions(pred_mcoptions)
    scores = {}
    for pred in pred_split:
        score = squad_em_f1(pred, gold_split)
        matched = score['matched_gold']
        f1 = score['F1']
        old_f1 = scores.get(matched, 0)
        if f1 > old_f1:
            scores[matched] = f1
    tot = sum(scores.values())
    return {"F1": tot / max(len(pred_split), 2)}


SCORING_SPECS = {
    "ruletaker_inferences": ruletaker_inferences_scores
}


def collate_scores(predictions):
    res_by_angle = {}
    metrics_by_angle = {}
    averaged_metrics = ['acc', 'EM', "F1", 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L',
                        'CIDEr', 'ROUGE_L_F', 'ROUGE_L_G', 'bad_parse', 'P', 'R',
                        'ROUGE_L_F_perfect_align', 'BLEURT', 'BLEURT_P', 'BLEURT_R', 'BLEURT_F1', 'BLEURT_perfect_align',
                        'BLEURT_acc', 'BLEURT_acc_perfect_align', 'acc_perfect_align', 'fraction_perfect_align',
                        'edited_or_not_acc',
                        'num_fever_queries', 'num_fever_queries_no_answer', 'num_fever_queries_no_rationale',
                        'num_premises', 'num_premises_valid', 'num_premises_valid_support', 'percent_valid_premises']
    aggregated_metrics = ['missing']
    for pred in predictions:
        angle = pred['angle_str']
        metrics = pred.get('metrics',{})
        if angle not in res_by_angle:
            res_by_angle[angle] = []
        res_by_angle[angle].append(pred)
        if angle not in metrics_by_angle:
            metrics_by_angle[angle] = {}
        metrics_by_angle[angle]['counter'] = metrics_by_angle[angle].get('counter', 0) + 1
        for slot, slot_metrics in metrics.items():
            if slot == "extra_slots":
                continue
            if slot not in metrics_by_angle[angle]:
                metrics_by_angle[angle][slot] = {}
            for metric in averaged_metrics + aggregated_metrics:
                if metric in slot_metrics:
                    metrics_by_angle[angle][slot][metric] = \
                        metrics_by_angle[angle][slot].get(metric, 0) + slot_metrics[metric]
    for angle, metrics in metrics_by_angle.items():
        counter = metrics['counter']
        for slot, slot_metrics in metrics.items():
            if not isinstance(slot_metrics, dict):
                continue
            for slot_metric, value in slot_metrics.items():
                if slot_metric in averaged_metrics:
                    slot_metrics[slot_metric] = value/counter
    return {"metrics_aggregated": metrics_by_angle, "by_angle": res_by_angle}


def score_aligned_entail_tree_proof(prediction, gold_list, angle, gold_json_record:dict, bleurt_scorer=None):
    res = {}
    if gold_list is None:
        res[angle] = {"missing_gold": 1}
        if prediction is None:
            res[angle]['missing'] = 1
        return res
    if prediction is None:
        res[angle] =  {"missing": 1}
        return res
    print(f"\n\n\n======================\n")
    print(f"pred:{prediction}")
    print(f"gold:{gold_list[0]}")
    print(f"\n\n\n======================\n")
    print(f"Reading predicted proof")
    sentences_pred, inferences_pred, int_to_all_ancestors_pred, relevant_sentences_pred, id_to_int_pred = \
        parse_entailment_step_proof(prediction, gold_json_record=gold_json_record)

    print(f"\n\n\n||||||||||||||||||||||\n")
    print(f"Reading gold proof")
    sentences_gold, inferences_gold, int_to_all_ancestors_gold, relevant_sentences_gold, id_to_int_gold = \
        parse_entailment_step_proof(gold_list[0], gold_json_record=gold_json_record)

    pred_int_to_gold_int_mapping, prediction_to_aligned_gold,  prediction_to_perfect_match = \
        align_conclusions_across_proofs(int_to_all_ancestors_pred, int_to_all_ancestors_gold,
                                        id_to_int_pred, id_to_int_gold)

    # res[angle+'-steps-unaligned'] = score_sentence_overlaps(sentences=sentences_pred, sentences_gold=sentences_gold)

    sentences_pred_aligned = rewrite_aligned_proof(prediction, pred_int_to_gold_int_mapping)
    print(f"\n\n\n++++++++++++++++++++++++++++++++++++")
    print(f"pred_int_to_gold_int_mapping:{pred_int_to_gold_int_mapping}")
    print(f"relevant_sentences_pred:{relevant_sentences_pred}")
    print(f"relevant_sentences_gold:{relevant_sentences_gold}")
    res[angle+'-leaves'] = score_sentence_overlaps(sentences=sorted(list(relevant_sentences_pred)),
                                               sentences_gold=sorted(list(relevant_sentences_gold)))

    res[angle + '-steps'] = score_sentence_overlaps(sentences=sorted(list(sentences_pred_aligned)),
                                                    sentences_gold=sorted(list(sentences_gold)))

    res[angle + '-steps']['pred_to_gold_mapping'] = pred_int_to_gold_int_mapping
    res[angle + '-steps']['sentences_pred_aligned'] = sentences_pred_aligned

    res[angle+'-intermediates'] = nlg_string_similarities_intermediates_with_F1(prediction_to_aligned_gold=prediction_to_aligned_gold,
                                                                                id_to_int_gold=id_to_int_gold,
                                                                                id_to_int_pred=id_to_int_pred,
                                                                        prediction_to_perfect_match=prediction_to_perfect_match,
                                                                        bleurt_scorer=bleurt_scorer)
    res[angle+'-overall'] = overall_proof_score(leaves=res[angle+'-leaves'],
                                                edges=res[angle+'-steps'],
                                                intermediates=res[angle+'-intermediates'])
    return res


def score_aligned_entail_tree_proof_onlyIR(prediction, gold_list, angle, gold_json_record:dict, pred_json_record: dict, bleurt_scorer=None):
    res = {}
    if gold_list is None:
        res[angle] = {"missing_gold": 1}
        if prediction is None:
            res[angle]['missing'] = 1
        return res
    if prediction is None:
        res[angle] =  {"missing": 1}
        return res
    print(f"\n\n++++++++++++++++++\nprediction:{prediction}")
    # print(f"pred_json_record:{pred_json_record}")
    sentences_pred, inferences_pred, int_to_all_ancestors_pred, relevant_sentences_pred, id_to_int_pred = \
        parse_entailment_step_proof_remove_ids(prediction, slot_json_record=pred_json_record)

    print(f"gold_json_record:{gold_json_record}")
    # print(f"gold_json_record:{gold_json_record}")
    sentences_gold, inferences_gold, int_to_all_ancestors_gold, relevant_sentences_gold, id_to_int_gold = \
        parse_entailment_step_proof_remove_ids(gold_list[0], slot_json_record=gold_json_record)

    print(f"^^^^^^^pred:{prediction}")
    print(f"========sentences_pred:{sentences_pred}")
    print(f"^^^^^^^gold:{gold_list[0]}")
    print(f"========sentences_gold:{sentences_gold}")

    print(f"Q: {pred_json_record['id']}")
    pred_int_to_gold_int_mapping, prediction_to_aligned_gold,  prediction_to_perfect_match = \
        align_conclusions_across_proofs(int_to_all_ancestors_pred, int_to_all_ancestors_gold,
                                        id_to_int_pred, id_to_int_gold)

    # res[angle+'-steps-unaligned'] = score_sentence_overlaps(sentences=sentences_pred, sentences_gold=sentences_gold)

    print(f"\n\n+++++++++++++++++++++++++\n")
    print(f"prediction:{prediction}")
    print(f"pred_int_to_gold_int_mapping:{pred_int_to_gold_int_mapping}")
    pred_sentences = pred_json_record['meta']['triples']
    pred_sentences['hypothesis'] = gold_json_record['hypothesis']
    sentences_pred_aligned, sentences_pred_aligned_strings = rewrite_aligned_proof_noids(prediction,
                                                                                         pred_int_to_gold_int_mapping,
                                                                                         pred_sentences=pred_sentences,
                                                                                         gold_ints=gold_json_record['meta']['intermediate_conclusions']
                                                                                         )
    res[angle+'-leaves'] = score_sentence_overlaps(sentences=sorted(list(relevant_sentences_pred)),
                                               sentences_gold=sorted(list(relevant_sentences_gold)))

    print(f"*********ID:{gold_json_record['id']}")
    print(f"*********sentences_pred_aligned:{sentences_pred_aligned_strings}")
    print(f"*********sentences_gold:{sentences_gold}")
    res[angle + '-steps'] = score_sentence_overlaps(sentences=sorted(list(sentences_pred_aligned_strings)),
                                                    sentences_gold=sorted(list(sentences_gold)))
    res[angle + '-steps']['pred_to_gold_mapping'] = pred_int_to_gold_int_mapping
    res[angle + '-steps']['sentences_pred_aligned'] = sentences_pred_aligned

    res[angle+'-intermediates'] = nlg_string_similarities_intermediates_with_F1(prediction_to_aligned_gold=prediction_to_aligned_gold,
                                                                                id_to_int_gold=id_to_int_gold,
                                                                                id_to_int_pred=id_to_int_pred,
                                                                                prediction_to_perfect_match=prediction_to_perfect_match,
                                                                        bleurt_scorer=bleurt_scorer)
    res[angle+'-overall'] = overall_proof_score(leaves=res[angle+'-leaves'],
                                                edges=res[angle+'-steps'],
                                                intermediates=res[angle+'-intermediates'])
    return res


def overall_proof_score(leaves, edges, intermediates):
    res = {}
    leaves_acc = leaves['acc']
    edges_acc = edges['acc']
    accuracy = leaves_acc * edges_acc * intermediates['BLEURT_acc']
    # accuracy = leaves_acc * edges_acc
    accuracy_align = leaves_acc * edges_acc * intermediates['BLEURT_acc_perfect_align']

    res['acc'] = accuracy
    res['acc_perfect_align'] = accuracy_align
    return res


def score_prediction_whole_proof(prediction, gold, prediction_json=None, dataset=None, scoring_spec=None, bleurt_scorer=None):
    angle = prediction.get('angle')
    if 'slots' in prediction:
        slots = prediction['slots']
    else:
        slots = decompose_slots(prediction['prediction'])
    answer_eval = "emf1"
    if scoring_spec is not None and "answer_eval" in scoring_spec:
        answer_eval = scoring_spec['answer_eval']
    elif "mcoptions" in gold:
        answer_eval = "mcq"
    elif dataset is not None and "narrative" in dataset:
        answer_eval = "nlg"

    hypothesis_eval = "nlg"
    if scoring_spec is not None and "hypothesis_eval" in scoring_spec:
        hypothesis_eval = scoring_spec['hypothesis_eval']

    proof_eval = "pn_eval"
    if scoring_spec is not None and "proof_eval" in scoring_spec:
        proof_eval = scoring_spec['proof_eval']

    res = {}
    angles_out = angle[1] if angle is not None else list(slots.keys())
    for angle in angles_out:
        gold_str = gold.get(angle)
        gold_list = [gold_str] if isinstance(gold_str, str) else gold_str
        slot = slots.get(angle)
        if angle == 'hypothesis':
            if hypothesis_eval == 'old_emf1':
                res[angle] = rc_answer_accuracy(slots, gold)
            elif hypothesis_eval == 'emf1':
                sentences_pred = slots.get('hypothesis').split(' , ')
                sentences_gold = gold['hypothesis'].split(' , ')
                res[angle] = score_sentence_overlaps(sentences=sentences_pred,
                                                     sentences_gold=sentences_gold)
            elif hypothesis_eval == 'mcq':
                res[angle] = mcq_answer_accuracy(slots, gold)
            elif hypothesis_eval == 'nlg':
                res[angle] = nlg_string_similarities(slot, gold_list)
            else:
                raise ValueError(f"Unknown answer_eval setting: {answer_eval}")
        elif angle in ['question', 'explanation']:
            res[angle] = nlg_string_similarities(slot, gold_list)
        elif angle in ['mcoptions']:
            res[angle] = rough_mcoptions_f1(slot, gold_str)
        elif angle in ['proof']:
            if proof_eval == "entail_whole_proof_align_eval":
                # This writes in multiple metrics
                res.update(score_aligned_entail_tree_proof(slot, gold_list, angle, gold_json_record=gold,bleurt_scorer=bleurt_scorer))
            elif proof_eval == "entail_whole_proof_align_eval_onlyIR":
                    # This writes in multiple metrics
                    res.update(score_aligned_entail_tree_proof_onlyIR(slot, gold_list, angle,
                                                                      gold_json_record=gold,
                                                                      pred_json_record=prediction_json,
                                                                      bleurt_scorer=bleurt_scorer))

            else:
                gold_proofs_list = []
                for gold_proof in gold_str.split(' OR '):
                    gold_parts = gold_proof.replace(', ', ',').split(',')
                    if len(gold_parts) == 2:
                        gold_proofs_list.append(gold_proof)
                        gold_proofs_list.append(f"{gold_parts[1]}, {gold_parts[0]}")
                res[angle] = squad_em_f1(slot, gold_proofs_list)
        elif angle in ['rationale']:
            pass  # Not implemented yet
        else:
            res[angle] = {}
    extras = []
    for slot in slots:
        if slot not in res:
            extras.append(slot)
    if len(extras) > 0:
        res['extra_slots'] = extras
    return res
