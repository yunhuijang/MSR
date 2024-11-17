
from evaluation import fingerprint_metrics, mol_translation_metrics, fcd_metric


architecture = 'multitask-text-and-chemistry-t5-small-standard'
run_name = 'chain-aromatic-con_ring_name-func_simple-chiral-iter'


file_name = f'predictions/two_stage_ft_cot/answer/{architecture}{run_name}.txt'

fcd_metric_score = fcd_metric.evaluate(file_name)
print(fcd_metric_score)