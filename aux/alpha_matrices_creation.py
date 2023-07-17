import ast
import pickle

import numpy as np
import pandas as pd
# Set the display options to show all columns
pd.set_option('display.max_columns', None)

# Read the CSV file into a DataFrame
df = pd.read_csv('similarity_comparisons_ratios.csv')
df = df[0:21]
# Initialize new columns for per-comparison alpha values (two per comparison)
df['alpha_0_2'] = 0
df['alpha_2_0'] = 0
df['alpha_0_1'] = 0
df['alpha_2_1'] = 0
# grab selected
df['alpha_1_0'] = 0
df['alpha_1_2'] = 0
comparisons_list = []

mask = pd.Series(False, index=df.index)
selection_values = [0, 1, 2]
# Iterate over three consecutive rows
for i in range(0, len(df), 3):
    group = df.iloc[i:i + 3]

    # Find the row with the maximum 'count_normalized' value
    max_row = group.loc[group['count_normalized'].idxmax()]

    selected0, selected1 = max_row['selected0'], max_row['selected1']
    negative_index = next(x for x in selection_values if x != selected0 and x != selected1)
    if selected0 == 0:
        ratio_anchor_negative = group.loc[(group['selected0'] == selected0) & (group['selected1'] ==
                                                                               negative_index)].iloc[0][
            'count_normalized']
        if selected1 == 1:
            ratio_positive_negative = group.loc[(group['selected0'] == selected1) & (group['selected1'] ==
                                                                                     negative_index)].iloc[0][
                'count_normalized']
        elif selected1 == 2:
            ratio_positive_negative = group.loc[(group['selected0'] == negative_index) & (group['selected1'] ==
                                                                                          selected1)].iloc[0][
                'count_normalized']
    elif selected0 == 1:
        if selected1 == 2:
            ratio_anchor_negative = group.loc[(group['selected0'] == negative_index) & (group['selected1'] ==
                                                                                        selected0)].iloc[0][
                'count_normalized']
            ratio_positive_negative = group.loc[(group['selected0'] == negative_index) & (group['selected1'] ==
                                                                                          selected1)].iloc[0][
                'count_normalized']
        else:
            assert False, "selected1 is not 1 or 2"
    else:
        assert False, "selected0 is not 0 or 1"

    alpha_selected0_selected1 = max_row['count_normalized'] - ratio_anchor_negative
    alpha_selected1_selected0 = max_row['count_normalized'] - ratio_positive_negative
    print(f"{alpha_selected0_selected1} . {alpha_selected1_selected0}")

    # Concatenate selected0 and selected1 to pattern match the alpha anchor_positive column
    alpha_anchor_positive_column = f"alpha_{selected0}_{selected1}"

    alpha_positive_anchor_column = f"alpha_{selected1}_{selected0}"

    df.loc[max_row.name, alpha_anchor_positive_column] = alpha_selected0_selected1
    df.loc[max_row.name, alpha_positive_anchor_column] = alpha_selected1_selected0
    comparisons_list.append(df.loc[max_row.name])

    # mask[group.index] = True
    #
    # # Assign 0 to the new columns for the remaining rows in the group
    # df.loc[~mask, [concatenated_values]] = 0
df = pd.DataFrame(comparisons_list)
df.reset_index(drop=True, inplace=True)
# Split the efforts_tuples values at the delimiter '_' and convert tokens to tuples
df['efforts_tuples'] = df['efforts_tuples'].apply(lambda x: [tuple(ast.literal_eval(token)) for token in x.split('_')])

# Print the modified DataFrame
print(df)

dict_similarity_classes_exemplars = pickle.load(open(
    "/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/exemplars_dir/tmp" \
                          "//similarity_labels_exemplars_dict.pickle", "rb"))
print(len(dict_similarity_classes_exemplars))
print([len(value) for key, value in dict_similarity_classes_exemplars.items()])
dict_label_to_id = {class_label:idx for idx, class_label in enumerate(dict_similarity_classes_exemplars.keys())}
print(dict_label_to_id)
n = len(dict_label_to_id)
zero_matrix = np.zeros((n, n))

for index, row in df.iterrows():
    efforts_tuple = row['efforts_tuples']
    row_index = efforts_tuple[0]
    col_index = efforts_tuple[1]
    count_normalized = row['count_normalized']

    # Assign count_normalized value to the indexed element in the numpy array
    zero_matrix[row_index, col_index] = count_normalized

print(zero_matrix)

