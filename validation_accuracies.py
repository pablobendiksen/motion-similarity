import pandas as pd
import conf
import os

path = conf.output_metrics_dir
path = "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/job_metrics/"
df = pd.DataFrame()

csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

for file in csv_files:
    df_temp = pd.read_csv(os.path.join(path, file))
    df = pd.concat([df, df_temp], ignore_index=True)

df = df.sort_values('Metric (MSE)', ascending=True)

df.to_csv(os.path.join(path, 'validation_accuracies.csv'), index=False)
