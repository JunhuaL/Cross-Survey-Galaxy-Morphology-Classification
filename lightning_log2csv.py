import glob
import pandas as pd
import sys
from tensorflow.python.summary.summary_iterator import summary_iterator

outputs_dir = sys.argv[1]
query = '**\\version*\\*\\events.out.tfevents*'
paths = glob.glob(outputs_dir+query,recursive=True)

results = dict()
for path in paths:
    new_path = path.replace(outputs_dir,'').split('\\')
    if results.get(new_path[0]):
        results[new_path[0]].append(path)
    else:
        results[new_path[0]] = [path]

csvs = []
for output in results:
    metric_paths = results[output]
    output_dict = dict()
    for x in range(0,len(metric_paths),2):
        metric_name = metric_paths[x].split('\\')[-2]
        values_1 = []
        values_2 = []
        for e in summary_iterator(metric_paths[x]):
            for v in e.summary.value:
                values_1.append(v.simple_value)

        for e in summary_iterator(metric_paths[x+1]):
            for v in e.summary.value:
                values_2.append(v.simple_value)
        output_dict[metric_name] = values_1 + values_2
    epochs = list(range(len(values_1))) + list(range(len(values_2)))
    output_dict['epoch'] = epochs
    df = pd.DataFrame(output_dict)
    df = df.set_index('epoch')
    df.to_csv(f'{output}.csv')

        