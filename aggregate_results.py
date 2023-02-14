from os import listdir
from os.path import isfile, join
import re
import numpy as np

dir_path = './stdouts/'

prefix = 'CIFAR100_wrn28x10_gradual_warm_scheduler0'
postfix = '.log'

dataset = prefix.split('_')[0]

log_files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
target_logs = [f for f in log_files if f.startswith(prefix) and f.endswith(postfix)]
# print(target_logs)

final_top1_scores = []
final_top5_scores = []

for log in target_logs:
    with open(join(dir_path,log)) as f:
        f = f.readlines()
    
    result_strs = []
    for line in f:
        if 'error rate' in line and dataset in line:
            result_strs.append(line)
    assert len(result_strs) == 1, print(result_strs)

    result_str = result_strs[0]
    top1_str = result_str.split('|')[1]
    top5_str = result_str.split('|')[2]
    
    top1_score = re.findall(r"\b[-+]?(?:\d*\.*\d+)", top1_str)
    # print(top1_score)
    assert len(top1_score) == 1, print(top1_str)
    top1_score = float(top1_score[0])

    top5_score = re.findall(r"\b[-+]?(?:\d*\.*\d+)", top5_str)
    assert len(top5_score) == 1, print(top5_str)
    top5_score = float(top5_score[0])

    final_top1_scores.append(top1_score)
    final_top5_scores.append(top5_score)

print(final_top1_scores)
print(final_top5_scores)
print()
mean_top1_score = sum(final_top1_scores) / len(final_top1_scores)
mean_top5_score = sum(final_top5_scores) / len(final_top5_scores)

print(f'Prefix: {prefix}')
print(f'Top1 Error rate. Mean: {round(mean_top1_score, 1)}, std: {round(np.std(final_top1_scores), 1)}')
print(f'Mean Top5 Error rate: {round(mean_top5_score, 1)}, std: {round(np.std(final_top5_scores), 1)}')
