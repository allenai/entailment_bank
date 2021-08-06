""" Evaluation script for EntailmentBank models. """

import argparse
import glob
import logging
import json
import os
import re
import sys
from bleurt import score

from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd()))  # noqa: E402 # isort:skip

from utils.angle_utils import decompose_slots, load_jsonl, save_json, shortform_angle, formatting
from utils.eval_utils import collate_scores, score_prediction_whole_proof

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, required=True, type=str,
                        help="Task name: task_1, task_2, task_3")
    parser.add_argument("--output_dir", default=None, required=True, type=str,
                        help="Directory to store scores.")
    parser.add_argument("--split", default=None, required=True, type=str, help="Which split (train/dev/test) to evaluate.")
    parser.add_argument("--prediction_file", default=None, required=True, type=str,
                        help="Prediction file(s) to score.")
    parser.add_argument("--bleurt_checkpoint", default="true", type=str,
                        help="Path to the BLEURT model checkpoint (Download from https://github.com/google-research/bleurt#checkpoints) "
                             "We use bleurt-large-512 model for EntailmentBank evaluation")

    args = parser.parse_args()
    return args


def split_info_sentences(context):
    words_list = context.split(" ")
    sentence_ids = re.findall(r'[\w\.]+[0-9]+:', context)
    sentence_dict = dict()
    prev_sid = ""
    prev_sentence_parts = []
    for word in words_list:
        if word in sentence_ids:
            if prev_sid:
                sentence_dict[prev_sid] = ' '.join(prev_sentence_parts)
                prev_sentence_parts = []
            prev_sid = word
        else:
            prev_sentence_parts.append(word)

    if prev_sid:
        sentence_dict[prev_sid] = ' '.join(prev_sentence_parts)
    return sentence_dict


def score_predictions(predictions_file, score_file, score_json_file, gold_file, angle_file=None, dataset=None, bleurt_checkpoint=""):
    if args.bleurt_checkpoint:
        bleurt_scorer = score.BleurtScorer(bleurt_checkpoint)
    else:
        bleurt_scorer = None

    gold_data = load_jsonl(gold_file)
    gold_by_id = {g['id']: g for g in gold_data}

    gold_train_file = gold_file.replace("dev", "train")
    gold_train_data = load_jsonl(gold_train_file)
    # gold_train_by_id = {g['id']: g for g in gold_train_data}
    train_context_dict = dict()
    train_answers_dict = dict()
    for g in gold_train_data:
        # print(f"{g['meta']['triples']}")
        context_dict = dict(g['meta']['triples'])
        for context in context_dict.values():
            train_context_dict[context] = 1
        train_answers_dict[g['answer']] = 1

    is_jsonl = predictions_file.endswith(".jsonl")
    if not is_jsonl:
        angle_data = load_jsonl(angle_file)
    scores = []
    sort_angle = False


    diagnostics_tsv = open(score_json_file+".pred.metrics.tsv", "w")

    num_dev_answers = 0
    num_dev_answers_seen_in_train_context = 0
    num_dev_answers_seen_in_train_answers = 0

    with open(score_file, "w") as score_file, open(predictions_file, "r") as preds_file:
        for line_idx, line in tqdm(enumerate(preds_file)):
            if is_jsonl:
                pred = json.loads(line.strip())
            else:
                pred = {'id': angle_data[line_idx]['id'],
                        'angle': angle_data[line_idx]['angle'],
                        'prediction': line.strip()}

            angle = pred['angle']
            angle_canonical = shortform_angle(angle, sort_angle=sort_angle)
            pred['angle_str'] = angle_canonical
            item_id = pred['id']

            # if item_id not in ['CSZ20680']:
            #     continue

            if item_id not in gold_by_id:
                continue
                raise ValueError(f"Missing id in gold data: {item_id}")
            slots = decompose_slots(pred['prediction'])

            pred['slots'] = slots

            num_dev_answers += 1
            # print(f"======= pred: {pred}")
            # print(f">>>>>>>>>>>> id:{item_id}")
            metrics = score_prediction_whole_proof(pred, gold_by_id[item_id], dataset,
                                                      scoring_spec={
                                                          "hypothesis_eval": "nlg",
                                                          "proof_eval": "entail_whole_proof_align_eval",
                                                          # "proof_eval": "entail_whole_polish_proof_align_eval",
                                                      },
                                                    bleurt_scorer=bleurt_scorer)

            pred['metrics'] = metrics
            score_file.write(json.dumps(pred) + "\n")
            id = angle_data[line_idx]['id']
            goldslot_record = gold_by_id[id]
            # print(f"goldslot_record:{goldslot_record}")
            question_before_json = ""
            if 'meta' in goldslot_record and 'question' in goldslot_record['meta']:
                question_before_json = goldslot_record['meta']['question']

            question_json = {}
            if 'meta' in goldslot_record and 'question' in goldslot_record['meta']:
                question_json = goldslot_record['meta']['question']

            question_json['gold_proofs'] = question_json.get('proofs', "")
            question_json['proofs'] = ""

            hypothesis_f1 = metrics.get('hypothesis', dict()).get('ROUGE_L_F', -1)
            question_json['ROUGE_L_F'] = hypothesis_f1

            sentences_dict = split_info_sentences(goldslot_record['context'])
            sentence_set = []
            for sid, sent in sentences_dict.items():
                sentence_set.append(f"{sid}: {sent}")
            sent_str = formatting(sentence_set)

            gold_triples = goldslot_record['meta']['triples']
            gold_ints = goldslot_record.get('meta', dict()).get('intermediate_conclusions', dict())
            gold_ints['hypothesis'] = goldslot_record['hypothesis']
            gold_triples.update(gold_ints)
            gold_proof_str = goldslot_record['proof']
            # if '; ' in gold_proof_str:
            if True:
                gold_proof_steps = gold_proof_str.split(';')

                gold_proof_str_list = []
                for step in gold_proof_steps:
                    step = step.strip()
                    if step.strip() and len(step.split(' -> '))==2:
                        print(f"step:{step}")

                        parts = step.split(' -> ')
                        lhs_ids = parts[0].split('&')
                        rhs = parts[1]
                        if rhs == "hypothesis":
                            rhs = f"hypothesis: {gold_triples['hypothesis']}"
                        for lid in lhs_ids:
                            lhs_id = lid.strip()
                            print(f"QID:{item_id}")
                            print(f"gold_triples:{gold_triples}")
                            print(f"step:{step}")
                            gold_proof_str_list.append(f"{lhs_id}: {gold_triples[lhs_id]} &")
                        gold_proof_str_list.append(f"-> {rhs}")
                        gold_proof_str_list.append(f"-----------------")
                gold_proof_str_to_output = formatting(gold_proof_str_list)

            pred_triples = goldslot_record['meta']['triples']
            pred_triples['hypothesis'] = goldslot_record['hypothesis']
            pred_proof_str = pred['slots'].get('proof', "")
            # print(f"^^^^^^^^^^^^^^^^^^pred_proof_str:{pred_proof_str}")
            # if '; ' in pred_proof_str:
            if True:
                pred_proof_steps = pred_proof_str.split(';')
                pred_proof_str_list = []
                # print(f"\n\n=================")
                # print(f"pred_proof_str:{pred_proof_str}")
                for step in pred_proof_steps:
                    step = step.strip()
                    if step.strip() and len(step.split(' -> '))==2:
                        print(f"step:{step}")
                        parts = step.split(' -> ')
                        lhs_ids = parts[0].split('&')
                        if ',' in parts[0]:
                            lhs_ids = parts[0].split(',')
                        rhs = parts[1]
                        if rhs == "hypothesis" or "hypothesis" in rhs:
                            rhs = f"hypothesis: {pred_triples['hypothesis']}"
                        else:
                            rhs_parts = rhs.split(":")
                            int_id = rhs_parts[0]
                            int_str = rhs_parts[1].strip()
                            pred_triples[int_id] = int_str
                        for lid in lhs_ids:
                            lhs_id = lid.strip()
                            pred_proof_str_list.append(f"{lhs_id}: {pred_triples.get(lhs_id, 'NULL')} &")
                        pred_proof_str_list.append(f"-> {rhs}")
                        pred_proof_str_list.append(f"-----------------")
                pred_proof_str_to_output = formatting(pred_proof_str_list)

            if '; ' in pred_proof_str:
                pred_proof_steps = pred_proof_str.split('; ')
                pred_step_list = []

                for step in pred_proof_steps:
                    if step.strip():
                        pred_step_list.append(f"{step}; ")
                pred_proof_str = formatting(pred_step_list)

            relevance_f1 = "-"
            relevance_accuracy = "-"
            if 'relevance' in metrics:
                relevance_f1 = metrics['relevance']['F1']
                relevance_accuracy = metrics['relevance']['acc']

            proof_acc = "-"
            proof_f1 = "-"
            proof_alignements = "-"
            if 'aligned_proof' in metrics:
                proof_acc = metrics['aligned_proof']['acc']
                proof_f1 = metrics['aligned_proof']['F1']
                proof_alignements = metrics['aligned_proof']['pred_to_gold_mapping']

            inference_type = "none"
            if "abduction" in id:
                inference_type = "abduction"
            elif "deduction" in id:
                inference_type = "deduction"

            num_distractors = 0
            fraction_distractors = 0.0
            num_context_sent = len(goldslot_record['meta']['triples'])
            if 'distractors' in goldslot_record['meta']:
                num_distractors = len(goldslot_record['meta']['distractors'])
                fraction_distractors = 1.0 * num_distractors / num_context_sent

            distractor_ids = goldslot_record['meta'].get('distractors', [])
            pred_to_gold_mapping = metrics['proof-steps']['pred_to_gold_mapping']
            pred_to_gold_mapping_str = ""
            for pred_int, gold_int in pred_to_gold_mapping.items():
                pred_to_gold_mapping_str += f"p_{pred_int} -> g_{gold_int} ;; "
            diagnostics_tsv.write(f"{id}"
                                  f"\t{sent_str}"
                                  f"\t{goldslot_record['question']}"
                                  f"\t{goldslot_record['answer']}"
                                  f"\t{goldslot_record['hypothesis']}"
                                  f"\t{gold_proof_str_to_output}"
                                  f"\t{pred_proof_str_to_output}"
                                  f"\t{' ;; '.join(metrics['proof-steps']['sentences_pred_aligned'])}"
                                  f"\t{pred_to_gold_mapping_str}"
                                  f"\t{metrics['proof-leaves']['P']*100}"
                                  f"\t{metrics['proof-leaves']['R']*100}"
                                  f"\t{metrics['proof-leaves']['F1']*100}"
                                  f"\t{metrics['proof-leaves']['acc']*100}"
                                  f"\t{metrics['proof-steps']['F1']*100}"
                                  f"\t{metrics['proof-steps']['acc']*100}"
                                  f"\t{metrics['proof-intermediates']['BLEURT_P']*100}"
                                  f"\t{metrics['proof-intermediates']['BLEURT_R']*100}"
                                  f"\t{metrics['proof-intermediates']['BLEURT_F1']*100}"
                                  f"\t{metrics['proof-intermediates']['BLEURT']}"
                                  f"\t{metrics['proof-intermediates']['BLEURT_acc']*100}"
                                  f"\t{metrics['proof-intermediates']['fraction_perfect_align']*100}"
                                  f"\t{metrics['proof-overall']['acc']*100}"                                  
                                  f"\t{num_distractors}"
                                  f"\t{num_context_sent}"
                                  f"\t{fraction_distractors}"
                                  f"\t{', '.join(distractor_ids)}"
                                  "\n")

            scores.append(pred)

    print("\n=================\n"
          "Percentage recall per gold proof depth\n"
          "Gold_proof_depth\t#Gold answers\t#Correct predictions\t%accuracy (recall)\t%Gold answers\t%Correct Predictions")

    print(f"=========================")
    print(f"num_dev_answers:{num_dev_answers}")
    print(f"num_dev_answers_seen_in_train_context:{num_dev_answers_seen_in_train_context}")
    print(f"num_dev_answers_seen_in_train_answers:{num_dev_answers_seen_in_train_answers}")

    return scores


# Sample command
# python multi_angle/run_scorer_mf_all_at_once.py --angle_data_dir /Users/bhavanad/research_data/ruletaker/missing_facts/data/angles/OWA_d3_run3 --output_dir /Users/bhavanad/research_data/ruletaker/missing_facts/data/scorings/OWA_d3_run2/OWA_d3_run1_on_d3 --slot_root_dir /Users/bhavanad/research_data/ruletaker/missing_facts/data/slots/ --slot_data_dir OWA_d3_run3-slots --split test --prediction_file /Users/bhavanad/research_data/ruletaker/missing_facts/data/predictions/OWA_d3_T5large/OWA_d3_run1.on_d3.15k.pred.test.tsv
def main(args):
    prediction_files = args.prediction_file
    if "," in prediction_files:
        prediction_files = prediction_files.split(",")
    elif os.path.isdir(prediction_files):
        dir_path = prediction_files
        prediction_files = [f"{dir_path}/{f}" for f in os.listdir(prediction_files) if re.match(r'.*_predictions', f)]
    else:
        prediction_files = glob.glob(prediction_files)


    prediction_files.sort()
    root_dir = "data/processed_data"
    angle_data_dir = f"{root_dir}/angles/{args.task}/"
    slot_data_dir = f"{root_dir}/slots/{args.task}-slots/"
    angle_base_name = os.path.basename(angle_data_dir)
    slot_file = os.path.join(slot_data_dir, args.split + '.jsonl')
    if not os.path.exists(slot_file):
        if args.split == 'val' and os.path.exists(os.path.join(slot_data_dir, "dev.jsonl")):
            slot_file = os.path.join(slot_data_dir, "dev.jsonl")
        else:
            raise ValueError(f"Slot data file {slot_file} does not exist!")
    predictions_jsonl_format = True
    for prediction_file in prediction_files:
        if not prediction_file.endswith(".jsonl"):
            predictions_jsonl_format = False
    angle_file = None
    # If predictions not in jsonl format, need angle data to get correct ids and angles
    if not predictions_jsonl_format:
        angle_file = os.path.join(angle_data_dir, args.split + '.jsonl')
        if not os.path.exists(angle_file):
            if args.split == 'val' and os.path.exists(os.path.join(angle_data_dir, "dev.jsonl")):
                slot_file = os.path.join(angle_data_dir, "dev.jsonl")
            else:
                raise ValueError(f"Angle data file {angle_file} does not exist!")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("Scoring the following files: %s", prediction_files)
    all_metrics_aggregated = {}

    split = args.split
    output_dir = args.output_dir
    bleurt_checkpoint = args.bleurt_checkpoint

    sys.argv = sys.argv[:1]

    for prediction_file in prediction_files:
        if not os.path.exists(prediction_file):
            logger.warning(f"  File not found: {prediction_file}")
            continue
        score_file_base = os.path.basename(prediction_file).replace("predictions", "").replace("prediction", "")
        score_file_base = f"scores-{angle_base_name}-{split}-{score_file_base}"
        score_file = os.path.join(output_dir, score_file_base)
        score_json_file = os.path.join(output_dir, score_file_base.replace("tsv", "jsonl"))
        logger.info(f"***** Scoring predictions in {prediction_file} *****")
        logger.info(f"   Gold data from: {slot_file}")
        logger.info(f"   Full output in: {score_file}")

        scores = score_predictions(
            predictions_file=prediction_file,
            score_file=score_file,
            score_json_file=score_json_file,
            gold_file=slot_file,
            angle_file=angle_file,
            dataset=angle_base_name,
            bleurt_checkpoint=bleurt_checkpoint)
        collated = collate_scores(scores)
        all_metrics_aggregated[score_file] = collated['metrics_aggregated']
        logger.info(f"    Aggregated metrics:")
        for key, val in collated['metrics_aggregated'].items():
            logger.info(f"       {key}: {val}")
        print(f"\n======================")
        colmns_str = '\t'.join([
                                # 'leave-P', 'leave-R',
                                'leave-F1',	'leaves-Acc',
                                'steps-F1',	'steps-Acc',
                                'int-BLEURT-F1', 'int-BLEURT-Acc',
                                #'int-BLEURT_align',	'int-BLEURT-Acc_align',
                                'overall-Acc','overall-Acc_align','int-fraction-align'])
        print(f"collated:{collated['metrics_aggregated']}")
        aggr_metrics = collated['metrics_aggregated']['QAHC->P']
        metrics_str = '\t'.join([
            # prediction_file,
            # str(round(aggr_metrics['proof-leaves']['P']*100.0, 2)),
            # str(round(aggr_metrics['proof-leaves']['R']*100.0, 2)),
            str(round(aggr_metrics['proof-leaves']['F1']*100.0, 2)),
            str(round(aggr_metrics['proof-leaves']['acc']*100.0, 2)),
            str(round(aggr_metrics['proof-steps']['F1']*100.0, 2)),
            str(round(aggr_metrics['proof-steps']['acc']*100.0, 2)),
            str(round(aggr_metrics['proof-intermediates']['BLEURT_F1'] * 100.0, 2)),
            str(round(aggr_metrics['proof-intermediates']['BLEURT_acc'] * 100.0, 2)),
            #str(round(aggr_metrics['proof-intermediates']['BLEURT_perfect_align'], 2)),
            #str(round(aggr_metrics['proof-intermediates']['BLEURT_acc_perfect_align']*100.0,2)),
            str(round(aggr_metrics['proof-overall']['acc']*100.0, 2)),
            #str(round(aggr_metrics['proof-overall']['acc_perfect_align']*100.0, 2)),
            #str(round(aggr_metrics['proof-intermediates']['fraction_perfect_align'] * 100.0, 2)),
        ])
        print(f"{colmns_str}")
        print(f"{metrics_str}")
    save_json(f"{score_json_file}.metrics.json", all_metrics_aggregated)

if __name__ == "__main__":
    args = get_args()
    main(args)