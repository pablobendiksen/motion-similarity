import time

import conf
import tensorflow as tf
import numpy as np
import pandas as pd
import ast
import pickle
from conf import BatchStrategy

# Set the display options to show all columns
pd.set_option('display.max_columns', None)


class TripletMining:
    _self = None

    def __new__(cls):
        if cls._self is None:
            cls._self = super(TripletMining, cls).__new__(cls)
        return cls._self

    def __init__(self):
        print("TripletMining init")
        self.dict_similarity_classes_exemplars = pickle.load(open(
            conf.exemplars_dir + conf.similarity_dict_file_name, "rb"))
        print(f"classes: {self.dict_similarity_classes_exemplars.keys()}")
        # self.dict_neutral_exemplar = self.dict_similarity_classes_exemplars.pop((0, 0, 0, 0))
        self.num_states_drives = len(self.dict_similarity_classes_exemplars.keys())
        # assert self.num_states_drives == 56, f"Incorrect number of states + drives: {self.num_states_drives}"
        (self.matrix_alpha_left_right_right_left, self.matrix_alpha_left_neut_neut_left,
         self.matrix_alpha_right_neut_neut_right) = (np.zeros((self.num_states_drives, self.num_states_drives)) for _
                                                     in range(3))
        # Generate boolean matrices for each anchor_positive permutation.
        (self.matrix_bool_left_right, self.matrix_bool_right_left, self.matrix_bool_left_neut,
         self.matrix_bool_neut_left, self.matrix_bool_right_neut, self.matrix_bool_neut_right) \
            = (np.full((self.num_states_drives, self.num_states_drives), 0) for _ in range(6))
        print(self.matrix_alpha_left_right_right_left.shape)
        print(self.matrix_bool_right_left.shape)
        self.tensor_dists_left_right_right_left = tf.Variable(
            initial_value=tf.zeros((self.num_states_drives, self.num_states_drives)))
        # Shape (num_states_drives,)
        self.tensor_dists_class_neut = tf.Variable(initial_value=tf.zeros((self.num_states_drives,)))
        self.neutral_embedding = tf.Variable(initial_value=tf.zeros((conf.embedding_size,)))
        self.pre_process_comparisons_data()

    def _extract_neutral_embedding(self, embeddings):
        """Assigns the neutral embedding from the network output, shape (embedding_size,), to instance attribute

                        Args:
                            embeddings: tensor of shape (batch_size, embed_dim)

                        Returns:
                            embeddings: tensor of shape (batch_size - 1, embed_dim)
                """
        self.neutral_embedding.assign(embeddings[0])
        return embeddings[1:]

    def calculate_left_right_distances(self, embeddings, squared=True):
        """Compute the 2D matrix of distances between all 56 class embeddings.

        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.

        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """
        print(f"embeddings shape: {tf.shape(embeddings)}")
        if not conf.bool_fixed_neutral_embedding:
            embeddings = self._extract_neutral_embedding(embeddings)
        # shape (batch_size, batch_size)
        dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

        # Get squared L2 norm for each embedding (each embedding's dot product with itself
        # shape (batch_size,)
        square_norm = tf.linalg.diag_part(dot_product)

        # Compute the pairwise squared euclidean distance matrix:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

        # Because of computation errors, some distances might be negative, so we put everything >= 0.0 (see unit test)
        distances = tf.maximum(distances, 0.0)

        if not squared:
            # The gradient of sqrt is infinite when x == 0.0 (eg: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = tf.cast(tf.equal(distances, 0.0), float)
            distances = distances + mask * 1e-16

            distances = tf.sqrt(distances)

            # Correct the epsilon added: set the distances of the mask to be exactly 0.0
            distances = distances * (1.0 - mask)

        # self.tensor_dists_left_right_right_left.assign(distances)
        if not conf.bool_fixed_neutral_embedding:
            tf.debugging.assert_shapes([(tf.shape(distances), (tf.TensorShape([conf.similarity_batch_size - 1,
                                                                               conf.similarity_batch_size - 1]),))])
        else:
            tf.debugging.assert_shapes([(tf.shape(distances), (tf.TensorShape([conf.similarity_batch_size,
                                                                               conf.similarity_batch_size]),))])
        return distances

    def calculate_class_neut_distances(self, embeddings, squared=True):
        """Calculate 1D tensor of either squared L2 norm or L2 norm of differences between class embeddings and the
        neutral embedding of shape (embedding_size,), resulting in tensor of shape (batch_size,).

                Args:
                    embeddings: tensor of shape (batch_size, embed_dim)

                Returns:
                    None
        """
        if squared:
            # Compute squared Euclidean distance between each class embedding and the neutral embedding
            self.tensor_dists_class_neut.assign(tf.reduce_sum(tf.square(embeddings - self.neutral_embedding), axis=1))
        else:
            # Compute Euclidean distance between each class embedding and the neutral embedding
            self.tensor_dists_class_neut.assign(tf.norm(embeddings - self.neutral_embedding, ord="euclidean", axis=1))

    def pre_process_comparisons_data(self):

        def verify_comparison_data():
            # Check that the Dataframe has no repeated efforts_tuples values
            seen_tuples = set()
            for index, row in df_comparisons.iterrows():
                hashable_list = tuple(row['efforts_tuples'])
                # Check if the hashable list is already in the set
                if hashable_list in seen_tuples:
                    assert False, f"Duplicate df_comparison efforts_tuples row at index {index}: {row['efforts_tuples']}"
                seen_tuples.add(hashable_list)
                # Check if efforts_left equals efforts_right
                assert row['efforts_tuples'][0] != row['efforts_tuples'][1], (f"efforts_left equals efforts_right at "
                                                                              f"index: {index}: "
                                                                              f"{row['efforts_tuples']}") \
 \
                    # Check that the Dataframe does not include the neutral as a similarity class
            assert (0, 0, 0, 0) not in seen_tuples, "neutral similarity class included in comparisons data"
            # Check that the Dataframe has the correct number of similarity classes
            assert len(seen_tuples) == self.num_states_drives, (f"incomplete similarity class count in comparisons "
                                                                f"data: {len(seen_tuples)}")

        # generate df_alphas where each row is a comparison between two similarity classes (and the neutral) and
        # contains the corresponding two out of six alpha values (each comparison has two alpha values, one for each
        # of the positives
        def _generate_df_alphas():
            comparisons_list = []
            selection_values = [0, 1, 2]
            # Initialize new columns for per-comparison alpha values (two values created per comparison)
            df_comparisons['alpha_0_2'] = 0
            df_comparisons['alpha_2_0'] = 0
            df_comparisons['alpha_0_1'] = 0
            df_comparisons['alpha_2_1'] = 0
            df_comparisons['alpha_1_0'] = 0
            df_comparisons['alpha_1_2'] = 0
            # Iterate over three consecutive rows
            # selected_0 is either 0 or 1 and selected_1 is either 1 or 2 (else we terminate)
            for i in range(0, len(df_comparisons), 3):
                group = df_comparisons.iloc[i:i + 3]

                # Find the row within a triplet with the maximum 'count_normalized' value, thereby establishing the
                # positive pair (i.e., selected0 and selected1
                max_row = group.loc[group['count_normalized'].idxmax()]

                max_selected_0, max_selected_1 = max_row['selected0'], max_row['selected1']
                negative_index = next(x for x in selection_values if x != max_selected_0 and x != max_selected_1)
                # find the anchor_positive ratio value under the cases in which anchor is each of the positive pair,
                # respectively, and positive is the negative class
                if max_selected_0 == 0:
                    ratio_positive_1_negative = \
                        group.loc[(group['selected0'] == max_selected_0) & (group['selected1'] ==
                                                                            negative_index)].iloc[0][
                            'count_normalized']
                    # means max_row selected_0 is 0. Thusly, 2 is the negative index; grab it from the corresponding row
                    if max_selected_1 == 1:
                        ratio_positive_2_negative = \
                            group.loc[(group['selected0'] == max_selected_1) & (group['selected1'] ==
                                                                                negative_index)].iloc[
                                0][
                                'count_normalized']
                    elif max_selected_1 == 2:
                        ratio_positive_2_negative = \
                            group.loc[(group['selected0'] == negative_index) & (group['selected1'] ==
                                                                                max_selected_1)].iloc[
                                0][
                                'count_normalized']

                    else:
                        assert False, "selected1 is not 1 or 2"

                elif max_selected_0 == 1:
                    if max_selected_1 == 2:
                        ratio_positive_1_negative = \
                            group.loc[(group['selected0'] == negative_index) & (group['selected1'] ==
                                                                                max_selected_0)].iloc[
                                0][
                                'count_normalized']
                        ratio_positive_2_negative = \
                            group.loc[(group['selected0'] == negative_index) & (group['selected1'] ==
                                                                                max_selected_1)].iloc[
                                0][
                                'count_normalized']
                    else:
                        assert False, "selected1 is not 1 or 2"
                else:
                    assert False, "selected0 is not 0 or 1"
                # generate the two possible alpha values for a comparison (i.e., treating selected0 as anchor versus
                # treating selected1 as anchor)
                alpha_positive_1_positive_2 = max_row['count_normalized'] - ratio_positive_1_negative
                alpha_positive_2_positive_1 = max_row['count_normalized'] - ratio_positive_2_negative
                if max_row['efforts_tuples'] == '[-1,-1,-1,0]_[0,-1,-1,1]':
                    print(f"{alpha_positive_1_positive_2} . {alpha_positive_2_positive_1}")

                # Concatenate selected0 and selected1 to pattern match the alpha anchor_positive column
                alpha_selected_0_selected_1_column = f"alpha_{max_selected_0}_{max_selected_1}"

                alpha_selected_1_selected_0_column = f"alpha_{max_selected_1}_{max_selected_0}"

                df_comparisons.loc[max_row.name, alpha_selected_0_selected_1_column] = alpha_positive_1_positive_2
                df_comparisons.loc[max_row.name, alpha_selected_1_selected_0_column] = alpha_positive_2_positive_1
                comparisons_list.append(df_comparisons.loc[max_row.name])

            df_alphas = pd.DataFrame(comparisons_list)
            df_alphas.reset_index(drop=True, inplace=True)
            # print(f'{df_alphas=}')
            df_alphas = df_alphas[:10]
            return df_alphas

        def _populate_alpha_matrices_and_masks():
            # Iterate over the rows of the comparisons DataFrame
            repeat_class_comparison_counter = 0
            equal_comparison_counter = 0
            bool_swap_left_right = False
            for index, row in df_alphas.iterrows():
                efforts_tuple = row['efforts_tuples']
                # enforce constraint that i < j always corresponds to left, right / i > j to right, left
                if dict_label_to_id[efforts_tuple[0]] > dict_label_to_id[efforts_tuple[1]]:
                    row['efforts_tuples'] = [efforts_tuple[1], efforts_tuple[0]]
                    bool_swap_left_right = True
                efforts_left = row['efforts_tuples'][0]
                efforts_right = row['efforts_tuples'][1]
                index_left = dict_label_to_id[efforts_left]
                index_right = dict_label_to_id[efforts_right]

                ### temporary fix for erroneous similarity class, and for self to self comparison, in comparisons
                ### DataFrame
                if efforts_left == (0, 0, 0, 0) or efforts_left == efforts_right:
                    repeat_class_comparison_counter += 1
                    continue
                # for any comparison, left_index < right_index
                # left, right indices indicate location for left, neut anchor_positive alpha with respect to Left,
                # Neutral Matrix (and right, neut anchor_positive alpha with respect to Right, Neutral Matrix)
                # whereas right, left indices indicate location for neut, left anchor_positive alpha and neut,
                # right anchor_positive alpha, respectively.
                if row['alpha_0_2'] != 0:
                    print("entered alpha_0_2")
                    left_right_alpha = row['alpha_0_2']
                    right_left_alpha = row['alpha_2_0']
                    if bool_swap_left_right:
                        left_right_alpha = row['alpha_2_0']
                        right_left_alpha = row['alpha_0_2']
                    self.matrix_alpha_left_right_right_left[index_left, index_right] = left_right_alpha
                    self.matrix_bool_left_right[index_left, index_right] = 1
                    self.matrix_alpha_left_right_right_left[index_right, index_left] = right_left_alpha
                    self.matrix_bool_right_left[index_right, index_left] = 1
                elif row['alpha_0_1'] != 0:
                    print("entered alpha_0_1")
                    left_neutral_alpha = row['alpha_0_1']
                    neutral_left_alpha = row['alpha_1_0']
                    if bool_swap_left_right:
                        left_neutral_alpha = row['alpha_2_1']
                        neutral_left_alpha = row['alpha_1_2']
                    self.matrix_alpha_left_neut_neut_left[index_left, index_right] = left_neutral_alpha
                    self.matrix_bool_left_neut[index_left, index_right] = 1
                    self.matrix_alpha_left_neut_neut_left[index_right, index_left] = neutral_left_alpha
                    self.matrix_bool_neut_left[index_right, index_left] = 1
                elif row['alpha_2_1'] != 0:
                    print(f"entered alpha_2_1 at indices: {index_left} , {index_right}")
                    right_neutral_alpha = row['alpha_2_1']
                    neutral_right_alpha = row['alpha_1_2']
                    if bool_swap_left_right:
                        right_neutral_alpha = row['alpha_0_1']
                        neutral_right_alpha = row['alpha_1_0']
                    self.matrix_alpha_right_neut_neut_right[index_left, index_right] = right_neutral_alpha
                    # print(f"{self.matrix_alpha_right_neut_neut_right[index_left, index_right]}")
                    self.matrix_bool_right_neut[index_left, index_right] = 1
                    self.matrix_alpha_right_neut_neut_right[index_right, index_left] = neutral_right_alpha
                    self.matrix_bool_neut_right[index_right, index_left] = 1
                # implies equal selection (or no selection) across all three pairs of a triplet
                else:
                    equal_comparison_counter += 1
                bool_swap_left_right = False

            # print(f'{self.matrix_alpha_right_neut_neut_right[16]}')
            # print(f'{len(self.matrix_alpha_right_neut_neut_right[16])}')
            # print(f'{self.matrix_bool_right_neut[16]}')
            # print(f'{len(self.matrix_bool_right_neut[16])}')
            # print(f'{repeat_class_comparison_counter=}')
            # print(f'{equal_comparison_counter=}')

        def _convert_np_matrices_to_tensors():
            matrices_list = [self.matrix_alpha_left_right_right_left, self.matrix_alpha_left_neut_neut_left,
                             self.matrix_alpha_right_neut_neut_right]
            (self.matrix_alpha_left_right_right_left, self.matrix_alpha_left_neut_neut_left,
             self.matrix_alpha_right_neut_neut_right) \
                = (tf.convert_to_tensor(matrix, dtype=tf.float32) for matrix in matrices_list)

            matrices_bool_list = [self.matrix_bool_left_right, self.matrix_bool_right_left, self.matrix_bool_left_neut,
                                  self.matrix_bool_neut_left, self.matrix_bool_right_neut, self.matrix_bool_neut_right]
            (self.matrix_bool_left_right, self.matrix_bool_right_left, self.matrix_bool_left_neut,
             self.matrix_bool_neut_left, self.matrix_bool_right_neut, self.matrix_bool_neut_right) \
                = (tf.convert_to_tensor(matrix, dtype=tf.float32) for matrix in matrices_bool_list)

        # read_in R generated similarity comparisons ratios csv
        df_comparisons = pd.read_csv('../aux/similarity_comparisons_ratios.csv')
        # Split the efforts_tuples values at the delimiter '_' and convert tokens to tuples
        df_comparisons['efforts_tuples'] = df_comparisons['efforts_tuples'].apply(
            lambda x: [tuple(ast.literal_eval(token)) for token in x.split('_')])
        ### (Temporary) Remove similarity classes from the dictionary that are not present in the comparisons DataFrame
        ### (e.g., (-1, -1, -1, 0), etc)
        set_comparison_classes = set([efforts_tuple[0] for efforts_tuple in df_comparisons['efforts_tuples']]).union(
            set([efforts_tuple[1] for efforts_tuple in df_comparisons['efforts_tuples']]))
        print(f"len set: {len(set_comparison_classes)}")
        dict_similarity_classes_exemplars = {key: value for key, value in
                                             self.dict_similarity_classes_exemplars.items() if
                                             key in
                                             set_comparison_classes}
        # Create a dictionary mapping the n similarity class labels to integers
        dict_label_to_id = {class_label: idx for idx, class_label in
                            enumerate(dict_similarity_classes_exemplars.keys())}
        dict_id_to_label = {idx: class_label for idx, class_label in
                            enumerate(dict_similarity_classes_exemplars.keys())}
        print(f"reduced dict label to id len: {len(dict_label_to_id)}")
        print(f"k,v of dict id to label: {dict_id_to_label.items()}")
        # verify_comparison_data()
        df_alphas = _generate_df_alphas()
        _populate_alpha_matrices_and_masks()
        _convert_np_matrices_to_tensors()
        print(f"type of matrix_bool_left_right: {type(self.matrix_bool_left_right)}")


# @tf.keras.register_keras_serializable()
# class TripletsPositiveLosses(tf.keras.metrics.Metric):
#     # A custom Keras metric to compute the number of positive losses across 1540 x 2 = 3080 triplets
#     def __init__(self, name="triplets_positives", **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.true_positives = self.add_weight(name="tp", initializer="zeros")
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
#         values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
#         values = tf.cast(values, "float32")
#         if sample_weight is not None:
#             sample_weight = tf.cast(sample_weight, "float32")
#             values = tf.multiply(values, sample_weight)
#         self.true_positives.assign_add(tf.reduce_sum(values))
#
#     def result(self):
#         return self.true_positives
#
#     def reset_state(self):
#         # The state of the metric will be reset at the start of each epoch.
#         self.true_positives.assign(0.0)


# consider alternative loss function based on cosine similarity:
# self.loss = tf.keras.losses.CosineSimilarity(axis=1)
# ap_distance = self.loss(anchor, positive)
# an_distance = self.loss(anchor, negative)
# loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
def calculate_triplet_loss(y_true, y_pred, triplet_mining_instance, batch_strategy=BatchStrategy.ALL):
    """Calculate 1D tensor of either squared L2 norm or L2 norm of differenc
    es between class embeddings and the
            neutral embedding of shape (embedding_size,), resulting in tensor of shape (batch_size,).

                    Args:
                        y_true: supposed 'labels' of the batch (i.e., class indexes), tensor of size (batch_size,
                        ) where each element is singleton tensor of an integer in the range [1, 56]
                        y_pred: embeddings, tensor of shape (batch_size, embed_dim)
                        batch_strategy: Enum value indicating the batch strategy to use for triplet mining

                    Returns:
                        losses: tensor of shape (triplet_mining.num_states_drives, triplet_mining.num_states_drives)
            """
    # must flatten y_true to tensor of ints in order to sort it
    y_true_flat = tf.reshape(y_true, [-1])
    sorted_indices = tf.argsort(y_true_flat)
    y_true = tf.gather(y_true, sorted_indices)
    y_pred = tf.gather(y_pred, sorted_indices)
    classes_distances = triplet_mining_instance.calculate_left_right_distances(y_pred)
    triplet_mining_instance.calculate_class_neut_distances(y_pred)

    # Reshape the 1D tensor into a 2D tensor with shape (1, 56) (will allow for broadcasting)
    row_dists_class_neut = tf.reshape(triplet_mining_instance.tensor_dists_class_neut,
                                      (1, triplet_mining_instance.num_states_drives))
    # Reshape the 1D tensor into a 2D tensor of shape (56, 1) (will allow for broadcasting)
    column_dists_class_neut = tf.reshape(triplet_mining_instance.tensor_dists_class_neut,
                                         (triplet_mining_instance.num_states_drives, 1))

    # case 1: left and right are positives

    # L = anchor
    diff_lr_ln = classes_distances - column_dists_class_neut

    tf.debugging.assert_shapes(
        [(tf.shape(diff_lr_ln), (tf.TensorShape([triplet_mining_instance.num_states_drives,
                                                 triplet_mining_instance.num_states_drives]),))],
        message="Tensor diff_lr_ln has incorrect shape")
    # consider only cases where diff_lr_ln > 0 (i.e., l_n - l_r > 0)
    if batch_strategy == BatchStrategy.HARD:
        diff_lr_ln = tf.maximum(diff_lr_ln, 0.0)
        diff_lr_rl_alpha = diff_lr_ln + triplet_mining_instance.matrix_alpha_left_right_right_left
        diff_lr_alpha = tf.multiply(diff_lr_rl_alpha, triplet_mining_instance.matrix_bool_left_right)
        # ensure no negative losses
        triplet_loss_L_R = diff_lr_alpha
        tf.debugging.assert_equal(triplet_loss_L_R, tf.maximum(triplet_loss_L_R, 0.0), message="Negative losses exist")
    # consider only cases where diff_lr_ln < 0 yet triplet_loss_L_N > 0 (i.e., l_r - l_n < 0 and l_r - l_n + alpha > 0)
    elif batch_strategy == BatchStrategy.SEMI_HARD:
        diff_lr_ln = tf.where(diff_lr_ln < 0, diff_lr_ln, 0)
        diff_lr_rl_alpha = diff_lr_ln + triplet_mining_instance.matrix_alpha_left_right_right_left
        diff_lr_alpha = tf.multiply(diff_lr_rl_alpha, triplet_mining_instance.matrix_bool_left_right)
        triplet_loss_L_R = tf.maximum(diff_lr_alpha, 0.0)
    # consider all cases where diff_lr_ln + alpha > 0 (i.e., dist(l_r) - dist(l_n) + alpha > 0)
    elif batch_strategy == BatchStrategy.ALL:
        diff_lr_rl_alpha = diff_lr_ln + triplet_mining_instance.matrix_alpha_left_right_right_left
        diff_lr_alpha = tf.multiply(diff_lr_rl_alpha, triplet_mining_instance.matrix_bool_left_right)
        triplet_loss_L_R = tf.maximum(diff_lr_alpha, 0.0)

    # R = anchor
    diff_rl_rn = classes_distances - row_dists_class_neut
    if batch_strategy == BatchStrategy.HARD:
        diff_rl_rn = tf.maximum(diff_rl_rn, 0.0)
        diff_rl_rn_alpha = diff_rl_rn + triplet_mining_instance.matrix_alpha_left_right_right_left
        diff_rl_alpha = tf.multiply(diff_rl_rn_alpha, triplet_mining_instance.matrix_bool_right_left)
        triplet_loss_R_L = diff_rl_alpha
        tf.debugging.assert_equal(triplet_loss_R_L, tf.maximum(triplet_loss_R_L, 0.0), message="Negative losses exist")
    elif batch_strategy == BatchStrategy.SEMI_HARD:
        diff_rl_rn = tf.where(diff_rl_rn < 0, diff_rl_rn, 0)
        diff_rl_rn_alpha = diff_rl_rn + triplet_mining_instance.matrix_alpha_left_right_right_left
        diff_rl_alpha = tf.multiply(diff_rl_rn_alpha, triplet_mining_instance.matrix_bool_right_left)
        triplet_loss_R_L = tf.maximum(diff_rl_alpha, 0.0)
    elif batch_strategy == BatchStrategy.ALL:
        diff_rl_rn_alpha = diff_rl_rn + triplet_mining_instance.matrix_alpha_left_right_right_left
        diff_rl_alpha = tf.multiply(diff_rl_rn_alpha, triplet_mining_instance.matrix_bool_right_left)
        triplet_loss_R_L = tf.maximum(diff_rl_alpha, 0.0)
    # case 2: left and neutral are positives

    # L = anchor
    diff_ln_lr = column_dists_class_neut - classes_distances
    tf.debugging.assert_shapes([(tf.shape(diff_ln_lr), (tf.TensorShape([triplet_mining_instance.num_states_drives,
                                                 triplet_mining_instance.num_states_drives]),))],
                               message="Tensor diff_ln_lr has incorrect shape")
    if batch_strategy == BatchStrategy.HARD:
        diff_ln_lr = tf.where(diff_ln_lr > 0, diff_ln_lr, 0)
        diff_ln_lr_alpha = diff_ln_lr + triplet_mining_instance.matrix_alpha_left_neut_neut_left
        # we care only for differences with corresponding alpha_left_neut values (i.e., relevant l_n cases)
        diff_ln_lr_alpha = tf.multiply(diff_ln_lr_alpha, triplet_mining_instance.matrix_bool_left_neut)
        # ensure no negative losses
        triplet_loss_L_N = diff_ln_lr_alpha
        tf.debugging.assert_equal(triplet_loss_L_N, tf.maximum(triplet_loss_L_N, 0.0), message="Negative losses exist")
    elif batch_strategy == BatchStrategy.SEMI_HARD:
        diff_ln_lr = tf.where(diff_ln_lr < 0, diff_ln_lr, 0)
        diff_ln_lr_alpha = diff_ln_lr + triplet_mining_instance.matrix_alpha_left_neut_neut_left
        diff_ln_lr_alpha = tf.multiply(diff_ln_lr_alpha, triplet_mining_instance.matrix_bool_left_neut)
        triplet_loss_L_N = tf.where(diff_ln_lr_alpha > 0, diff_ln_lr_alpha, 0)
    elif batch_strategy == BatchStrategy.ALL:
        diff_ln_lr_alpha = diff_ln_lr + triplet_mining_instance.matrix_alpha_right_neut_neut_right
        diff_ln_lr_alpha = tf.multiply(diff_ln_lr_alpha, triplet_mining_instance.matrix_bool_left_neut)
        triplet_loss_L_N = tf.where(diff_ln_lr_alpha > 0, diff_ln_lr_alpha, 0)
    # N = anchor
    # diff_nl_nr = column_dists_class_neut - row_dists_class_neut
    diff_nl_nr = row_dists_class_neut - column_dists_class_neut
    tf.debugging.assert_shapes([(tf.shape(diff_nl_nr), (tf.TensorShape([triplet_mining_instance.num_states_drives,
                                                 triplet_mining_instance.num_states_drives]),))],
                               message="Tensor diff_nl_nr has incorrect shape")
    if batch_strategy == BatchStrategy.HARD:
        diff_nl_nr = tf.where(diff_nl_nr > 0, diff_nl_nr, 0)
        diff_nl_nr_alpha = diff_nl_nr + triplet_mining_instance.matrix_alpha_left_neut_neut_left
        diff_nl_nr_alpha = tf.multiply(diff_nl_nr_alpha, triplet_mining_instance.matrix_bool_neut_left)
        triplet_loss_N_L = diff_nl_nr_alpha
        tf.debugging.assert_equal(triplet_loss_N_L, tf.maximum(triplet_loss_N_L, 0.0), message="Negative losses exist")
    elif batch_strategy == BatchStrategy.SEMI_HARD:
        diff_nl_nr = tf.where(diff_nl_nr < 0, diff_nl_nr, 0)
        diff_nl_nr_alpha = diff_nl_nr + triplet_mining_instance.matrix_alpha_left_neut_neut_left
        diff_nl_nr_alpha = tf.multiply(diff_nl_nr_alpha, triplet_mining_instance.matrix_bool_neut_left)
        triplet_loss_N_L = tf.where(diff_nl_nr_alpha > 0, diff_nl_nr_alpha, 0)
    elif batch_strategy == BatchStrategy.ALL:
        diff_nl_nr_alpha = diff_nl_nr + triplet_mining_instance.matrix_alpha_left_neut_neut_left
        diff_nl_nr_alpha = tf.multiply(diff_nl_nr_alpha, triplet_mining_instance.matrix_bool_neut_left)
        triplet_loss_N_L = tf.where(diff_nl_nr_alpha > 0, diff_nl_nr_alpha, 0)

    ### case 3: right and neutral are positives

    # R = anchor
    diff_rn_rl = row_dists_class_neut - tf.transpose(classes_distances)
    if batch_strategy == BatchStrategy.HARD:
        diff_rn_rl = tf.where(diff_rn_rl > 0, diff_rn_rl, 0)
        diff_r_n_r_l_alpha = diff_rn_rl + triplet_mining_instance.matrix_alpha_right_neut_neut_right
        diff_r_n_r_l_alpha = tf.multiply(diff_r_n_r_l_alpha, triplet_mining_instance.matrix_bool_right_neut)
        # remove negative losses
        triplet_loss_R_N = diff_r_n_r_l_alpha
        tf.debugging.assert_equal(triplet_loss_R_N, tf.maximum(triplet_loss_R_N, 0.0), message="Negative losses exist")
    elif batch_strategy == BatchStrategy.SEMI_HARD:
        diff_rn_rl = tf.where(diff_rn_rl < 0, diff_rn_rl, 0)
        diff_r_n_r_l_alpha = diff_rn_rl + triplet_mining_instance.matrix_alpha_right_neut_neut_right
        diff_r_n_r_l_alpha = tf.multiply(diff_r_n_r_l_alpha, triplet_mining_instance.matrix_bool_right_neut)
        triplet_loss_R_N = tf.where(diff_r_n_r_l_alpha > 0, diff_r_n_r_l_alpha, 0)
    elif batch_strategy == BatchStrategy.ALL:
        diff_r_n_r_l_alpha = diff_rn_rl + triplet_mining_instance.matrix_alpha_right_neut_neut_right
        diff_r_n_r_l_alpha = tf.multiply(diff_r_n_r_l_alpha, triplet_mining_instance.matrix_bool_right_neut)
        triplet_loss_R_N = tf.where(diff_r_n_r_l_alpha > 0, diff_r_n_r_l_alpha, 0)
    # check if diff_r_n_r_l_alpha and diff_rn_rl contain non-zero values in the same locations
    # condition = tf.math.logical_and(tf.math.not_equal(diff_rn_rl, 0), tf.math.not_equal(diff_r_n_r_l_alpha, 0))
    # tf.debugging.Assert(tf.math.reduce_all(condition),
    #                     message="Difference rn_rl tensors do not have non-zero elements in the same cells")
    tf.debugging.assert_equal(triplet_loss_R_N,
                              tf.multiply(triplet_loss_R_N, triplet_mining_instance.matrix_bool_right_neut),
                              message="Improper triplet_loss_R_N generation")
    # N = anchor
    diff_nr_nl = column_dists_class_neut - row_dists_class_neut
    tf.debugging.assert_shapes(
        [(diff_nr_nl, (triplet_mining_instance.num_states_drives, triplet_mining_instance.num_states_drives))],
        message="Tensor diff_nr_nl has incorrect shape")
    if batch_strategy == BatchStrategy.HARD:
        diff_nr_nl_alpha = diff_nr_nl + triplet_mining_instance.matrix_alpha_right_neut_neut_right
        diff_nr_nl_alpha = tf.multiply(diff_nr_nl_alpha, triplet_mining_instance.matrix_bool_neut_right)
        triplet_loss_N_R = diff_nr_nl_alpha
        tf.debugging.assert_equal(triplet_loss_N_R, tf.maximum(triplet_loss_N_R, 0.0), message="Negative losses exist")
    elif batch_strategy == BatchStrategy.SEMI_HARD:
        diff_nr_nl_alpha = diff_nr_nl + triplet_mining_instance.matrix_alpha_right_neut_neut_right
        diff_nr_nl_alpha = tf.multiply(diff_nr_nl_alpha, triplet_mining_instance.matrix_bool_neut_right)
        triplet_loss_N_R = tf.where(diff_nr_nl_alpha > 0, diff_nr_nl_alpha, 0)
    elif batch_strategy == BatchStrategy.ALL:
        diff_nr_nl_alpha = diff_nr_nl + triplet_mining_instance.matrix_alpha_right_neut_neut_right
        diff_nr_nl_alpha = tf.multiply(diff_nr_nl_alpha, triplet_mining_instance.matrix_bool_neut_right)
        triplet_loss_N_R = tf.where(diff_nr_nl_alpha > 0, diff_nr_nl_alpha, 0)

    losses = (triplet_loss_L_R + triplet_loss_R_L + triplet_loss_L_N + triplet_loss_N_L + triplet_loss_R_N +
              triplet_loss_N_R)
    # assertion check that no negative losses exist
    tf.debugging.assert_equal(losses, tf.maximum(losses, 0.0), message="Negative losses exist")
    tf.debugging.assert_shapes([(tf.shape(losses), (tf.TensorShape([triplet_mining_instance.num_states_drives,
                                          triplet_mining_instance.num_states_drives]),))],
                               message="Comparisons loss tensor has incorrect shape")
    return losses


#  custom loss function, a wrapper for calculate_triplet_loss, passed to Keras' compile method, that computes a triplet
#  loss for a batch of embeddings
def batch_triplet_loss(y_true, y_pred):
    """Build triplet loss over a batch of embeddings.

        We calculate triplet losses for all anchor-positive possibilities, and mask for semi-hard cases only.

        Args:
            y_true: supposed 'labels' of the batch (i.e., class indexes), tensor of size (batch_size,)
            y_pred: embeddings, tensor of shape (batch_size, embed_dim)

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
    triplet_mining = TripletMining()
    # print(f"y_true numpy shape: {y_true.numpy().shape}")
    # print(f"y_pred numpy shape: {y_pred.numpy().shape}")
    print(f"type of y_true: {type(y_true)}")
    print(f"type of y_pred: {type(y_pred)}")
    # tf.print(f"y_true: {y_true}")
    print(f"y_pred shape: {tf.shape(y_pred)}")

    triplet_loss = calculate_triplet_loss(y_true, y_pred, triplet_mining)
    # Count number of positive err triplets (where triplet_loss > 0)
    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), float)
    num_positive_triplets = tf.reduce_sum(valid_triplets)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
    tf.debugging.assert_scalar(triplet_loss, message="Epoch loss is not a scalar")

    return triplet_loss

# possible eval metrics: counting the triplets for which the positive distance (anchor - positive) is less than
# the negative distance (anchor - negative) (by at least the margin) and then dividing by the total number of
# triplets in the batch (i.e., proportion of zero loss triplets)

# classification: model embeddings to classify motions (take an unseen motion exemplar (representing one class) and
# compare it - using L2 normalized Euclidean distance - with its nearest neighbor.

# ranking accuracy:
# compute L2 normalized distances between all pairs of classes in the embedding space, and generate Spearman
# rank correlation coefficient (SROCC) between the sorted distances (ascending order) and the complement of the
# user normalized similarity scores (i.e., 1 - normalized similarity scores).
