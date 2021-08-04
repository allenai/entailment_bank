# entailment_bank

Example commands:

Task1:
python eval/run_scorer.py --angle_data_dir data/processed_data/angles/task_1/ --slot_root_dir data/processed_data/slots --slot_data_dir task_1-slots --split test --prediction_file data/processed_data/predictions/task_1/run1.pred.test.tsv  --output_dir  data/processed_data/scorings/task_1/on_test --scramble_slots=False --bleurt_checkpoint ""

Task2:
python eval/run_scorer_entail_trees_whole_proof.py --angle_data_dir data/processed_data/angles/task_2/ --slot_root_dir data/processed_data/slots --slot_data_dir task_2-slots --split test --prediction_file data/processed_data/predictions/task_2/run1.pred.test.tsv  --output_dir  data/processed_data/scorings/task_2/on_test --scramble_slots=False --bleurt_checkpoint ""

