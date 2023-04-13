import pandas as pd
import conf
import os

path = "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/motion-similarity/job_model_accuracies"
df = pd.DataFrame()

csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

for file in csv_files:
    df_temp = pd.read_csv(os.path.join(path, file))
    df = pd.concat([df, df_temp], ignore_index=True)

df = df.sort_values('Val Accuracy', ascending=False)

df.to_csv(os.path.join(path, 'validation_accuracies.csv'), index=False)
