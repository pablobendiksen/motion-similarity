"""
static module for organizing triplet mining data as well as performing online triplet mining
"""
from pathlib import Path
import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)
import numpy as np
import pandas as pd
import ast
import pickle


class TripletMining:
    def __init__(self, bool_drop, bool_fixed, squared_left_right, squared_class_neut, anim_name, config):
        self.config = config
        self.dict_similarity_classes_exemplars = {}
        self.matrix_alpha_left_right_right_left = None
        self.matrix_alpha_left_neut_neut_left = None
        self.matrix_alpha_right_neut_neut_right = None
        self.matrix_bool_left_right = None
        self.matrix_bool_right_left = None
        self.matrix_bool_left_neut = None
        self.matrix_bool_neut_left = None
        self.matrix_bool_right_neut = None
        self.matrix_bool_neut_right = None

        self.num_states_drives = 0
        self.tensor_dists_left_right_right_left = None
        self.tensor_dists_class_neut = None
        self.neutral_embedding = None

        self.bool_drop_neutral_exemplar = bool_drop
        self.bool_fixed_neutral_embedding = bool_fixed
        self.squared_left_right_euc_dist = squared_left_right
        self.squared_class_neut_dist = squared_class_neut
        self.batch_size = self.config.similarity_per_anim_class_num
        print(f"TripletMining: batch size: {self.batch_size}")
        self.initialize_triplet_mining(anim_name)

    def initialize_triplet_mining(self, anim_name):
        """
        Initialize the triplet mining module's state variables.

        This function loads necessary data, sets up state variables, and performs preprocessing.

        Args:
            anim_name: str: name of the animation (e.g., "walking", "pointing", "picking")

        Returns:
            None
        """

        print("Initializing Triplet Mining module state variables")
        self.dict_similarity_classes_exemplars = pickle.load(open(
            self.config.similarity_exemplars_dir + anim_name + "_" + self.config.similarity_dict_file_name, "rb"))
        print(f"classes: {self.dict_similarity_classes_exemplars.keys()}")

        key_to_remove = (0, 0, 0, 0)
        if key_to_remove not in self.dict_similarity_classes_exemplars:
            assert False, f"triplet_mining.py: Key '{key_to_remove}' not found in dict_similarity_classes_exemplars"
        if self.bool_drop_neutral_exemplar:
            _removed_value = self.dict_similarity_classes_exemplars.pop(key_to_remove)
            print(f"Removed key '{key_to_remove}' from dict_similarity_classes_exemplars")
            self.num_states_drives = len(self.dict_similarity_classes_exemplars.keys())
        else:
            self.num_states_drives = len(self.dict_similarity_classes_exemplars.keys()) - 1
        print(f"triplet_mining:init: loaded: {self.num_states_drives} states + drives")
        (self.matrix_alpha_left_right_right_left, self.matrix_alpha_left_neut_neut_left,
         self.matrix_alpha_right_neut_neut_right,
         self.matrix_bool_left_right, self.matrix_bool_right_left, self.matrix_bool_left_neut,
         self.matrix_bool_neut_left,
         self.matrix_bool_right_neut, self.matrix_bool_neut_right) = [tf.Variable(
            initial_value=tf.zeros((self.num_states_drives, self.num_states_drives), dtype=tf.float32)) for _ in range(9)]
        self.tensor_dists_class_neut = tf.Variable(initial_value=tf.zeros((self.num_states_drives,)))
        self.neutral_embedding = tf.Variable(initial_value=tf.zeros((self.config.embedding_size,)))
        self.subset_global_dict()
        self.pre_process_comparisons_data(anim_name)

    def subset_global_dict(self):
        """
        Subsets the global dictionary of similarity classes based on data in the comparisons DataFrame.

        Args:
            None

        Returns:
            None
        """
        dict_label_to_id = {class_label: idx for idx, class_label in
                            enumerate(self.dict_similarity_classes_exemplars.keys())}
        print(dict_label_to_id)

    # def extract_neutral_embedding(embeddings):
    #     """
    #     Assigns the neutral embedding from the network output, shape (embedding_size,), to instance attribute
    #
    #     Args:
    #         embeddings: tensor of shape (batch_size, embed_dim)
    #
    #     Returns:
    #         embeddings: tensor of shape (batch_size - 1, embed_dim)
    #     """
    #     # print(f'extract_neutral_embedding() called with embeddings of shape: {tf.shape(embeddings)}, neutral is: {embeddings[0]}')
    #     if not conf.bool_fixed_neutral_embedding:
    #         neutral_embedding.assign(embeddings[0])
    #         embeddings = embeddings[1:]
    #     return embeddings

    def zero_out_neutral_embedding(self, embeddings):
        if self.bool_drop_neutral_exemplar:
            modified_embeddings = embeddings
        else:
            modified_embeddings = embeddings[1:]

        return self.neutral_embedding, modified_embeddings

    def maintain_dynamic_neutral_embedding(self, embeddings):
        # print(f"Embeddings shape: {embeddings.shape}")  # Debugging line
        # print(f"Embeddings type: {type(embeddings)}")
        if self.bool_drop_neutral_exemplar:
            assert False, "triplet_mining.py: maintain_dynamic_neutral_embedding() called with bool_drop_neutral_exemplar set to True"
        # Assign the first embedding to neutral_embedding
        neutral_embedding = embeddings[0]

        # Remove the first element from the tensor
        modified_embeddings = embeddings[1:]

        return neutral_embedding, modified_embeddings

    def calculate_left_right_distances(self, embeddings):
        """Compute the 2D matrix of distances between all 56 class embeddings.

        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.

        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """

        # print(f"calculate_left_right_distances bool: {self.squared_left_right_euc_dist}")
        if self.bool_fixed_neutral_embedding:
            _neutral_embedding, modified_embeddings = self.zero_out_neutral_embedding(embeddings)
        else:
            _neutral_embedding, modified_embeddings = self.maintain_dynamic_neutral_embedding(embeddings)

        # shape (batch_size, batch_size)
        dot_product = tf.matmul(modified_embeddings, tf.transpose(modified_embeddings))
        # print(f"calculate_left_right_distances(): dot_product shape: {dot_product.shape}")

        # Get squared L2 norm for each embedding (each embedding's dot product with itself
        # shape (batch_size,)
        square_norm = tf.linalg.diag_part(dot_product)

        # Compute the pairwise squared euclidean distance matrix:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size-1, batch_size-1)
        distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

        # Because of computation errors, some distances might be negative, so we put everything >= 0.0 (see unit test)
        distances = tf.maximum(distances, 0.0)

        if not self.squared_left_right_euc_dist:
            # Compute pairwise Euclidean distances directly
            # shape (batch_size, batch_size)

            # distances = tf.norm(
            #     tf.expand_dims(modified_embeddings, axis=1) - tf.expand_dims(modified_embeddings, axis=0),
            #     axis=-1
            # )
            # The gradient of sqrt is infinite when x == 0.0 (eg: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = tf.cast(tf.equal(distances, 0.0), float)
            distances = distances + mask * 1e-16

            distances = tf.clip_by_value(distances, 1e-16, tf.reduce_max(distances))

            distances = tf.sqrt(distances)

            # Correct the epsilon added: set the distances of the mask to be exactly 0.0
            distances = distances * (1.0 - mask)

            tf.debugging.check_numerics(distances, "NaN or Inf values found in distances")

            # tf.debugging.assert_shapes([(tf.shape(distances), (tf.TensorShape([conf.similarity_batch_size,
            #                                                                    conf.similarity_batch_size]),))])

        # if not conf.bool_fixed_neutral_embedding:
        #     tf.debugging.assert_shapes([(tf.shape(distances), (tf.TensorShape([conf.similarity_batch_size - 1,
        #                                                                        conf.similarity_batch_size - 1]),))])
        # else:
        #     tf.debugging.assert_shapes([(tf.shape(distances), (tf.TensorShape([conf.similarity_batch_size,
        #                                                                        conf.similarity_batch_size]),))])
        return distances


    def calculate_class_neut_distances(self, embeddings):
        """
        Calculate 1D tensor of either squared L2 norm, or L2 norm, of differences between class embeddings and the
        neutral embedding.

        Args:
            embeddings: Tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, calculate squared L2 norm; if false, calculate L2 norm.

        Returns:
            None
        """
        # print(f"triplet_mining:calculate_class_neut_distances() bool: {self.squared_class_neut_dist}")
        if self.bool_fixed_neutral_embedding:
            neutral_embedding, modified_embeddings = self.zero_out_neutral_embedding(embeddings)
        else:
            neutral_embedding, modified_embeddings = self.maintain_dynamic_neutral_embedding(embeddings)
        if self.squared_class_neut_dist:
            # Compute squared Euclidean distance between each class embedding and the neutral embedding
            self.tensor_dists_class_neut.assign(tf.reduce_sum(tf.square(modified_embeddings - neutral_embedding), axis=1))
            # self.tensor_dists_class_neut.assign(tf.square(tf.norm(modified_embeddings - neutral_embedding, ord="euclidean", axis=1)))
        else:
            # Compute Euclidean distance between each class embedding and the neutral embedding
            self.tensor_dists_class_neut.assign(tf.norm(modified_embeddings - neutral_embedding, ord="euclidean", axis=1))

            # distances = modified_embeddings - neutral_embedding
            # squared_dists = tf.reduce_sum(tf.square(distances), axis=1)
            #
            # mask = tf.cast(tf.equal(squared_dists, 0.0), float)
            # squared_dists = squared_dists + mask * 1e-16
            # squared_dists = tf.clip_by_value(squared_dists, 1e-16, tf.reduce_max(squared_dists))
            # distances = tf.sqrt(squared_dists)
            # distances = distances * (1.0 - mask)
            #
            # tensor_dists_class_neut.assign(distances)

            # squared_dists = tf.reduce_sum(tf.square(modified_embeddings - neutral_embedding), axis=1)
            # euclidean_dists = tf.sqrt(squared_dists)
            # tensor_dists_class_neut.assign(euclidean_dists)

        # print(
        #     f"triplet_mining:calculate_class_neut_distances(), tensor_dists_class_neut shape: {tf.shape(self.tensor_dists_class_neut)}")

    def pre_process_comparisons_data(self, anim_name):
        """
        Preprocess user comparison data and populate alpha matrices and masks based on the data.

        Args:
            anim_name: str: name of the animation (e.g., "walking", "pointing", "picking")

        Returns:
            None
        """

        def verify_comparison_data():
            # Check that the Dataframe has no repeated efforts_tuples values
            seen_tuples = set()
            for index, row in self.df_comparisons.iterrows():
                hashable_list = tuple(row['efforts_tuples'])
                # Check if the hashable list is already in the set
                if hashable_list in seen_tuples:
                    assert False, f"Duplicate df_comparison efforts_tuples row at index {index}: {row['efforts_tuples']}"
                seen_tuples.add(hashable_list)
                # Ensure efforts_left does not equal efforts_right
                assert row['efforts_tuples'][0] != row['efforts_tuples'][1], (f"efforts_left equals efforts_right at "
                                                                              f"index: {index}: "
                                                                              f"{row['efforts_tuples']}")
            # Check that the Dataframe does not include the neutral as a similarity class
            assert (0, 0, 0, 0) not in seen_tuples, "neutral similarity class included in comparisons data"
            # Check that the Dataframe has the correct number of similarity classes
            assert len(seen_tuples) == self.num_states_drives, (f"incomplete similarity class count in comparisons "
                                                           f"data: {len(seen_tuples)}")

        def _generate_df_alphas():
            """
            generate alpha_dataframes where each row is a comparison between two similarity classes (and the neutral) and
            contains the corresponding two out of six alpha values (each comparison has two alpha values, one for each
            of the positives.

            df_comparisons: DataFrame: contains the user comparison data:
            columns - efforts_tuples, selected0, selected1, count, count_normalized, selected_motions

           Args:
               None

           Returns:
               None
            """
            counter_df_comparisons_triplets = 0
            comparisons_list = []
            selection_values = [0, 1, 2]
            # Initialize new columns for pairwise comparison alpha values (two values created per pairwise comparison)
            df_comparisons['alpha_0_2'] = 0
            df_comparisons['alpha_2_0'] = 0
            df_comparisons['alpha_0_1'] = 0
            df_comparisons['alpha_2_1'] = 0
            df_comparisons['alpha_1_0'] = 0
            df_comparisons['alpha_1_2'] = 0
            # Iterate over three consecutive rows
            # selected_0 is either 0 (agent left) or 1 (neutral) and selected_1 is either 1 or 2 (agent right) (else we terminate)
            for i in range(0, len(df_comparisons), 3):
                group = df_comparisons.iloc[i:i + 3]
                # counter_df_comparisons_triplets += 1
                # print(f"counter_df_comparisons_triplets: {counter_df_comparisons_triplets}")

                # Find the row within a triplet with the maximum 'count_normalized' value, thereby establishing the
                # positive pair (i.e., selected0 and selected1 which could be (0,1), (0,2) or (1,2)
                max_row = group.loc[group['count_normalized'].idxmax()]

                max_selected_0, max_selected_1 = max_row['selected0'], max_row['selected1']
                negative_index = next(x for x in selection_values if x != max_selected_0 and x != max_selected_1)
                # find the anchor_positive ratio value under the cases in which anchor is each of the positive pair,
                # respectively, and positive is the negative class
                if max_selected_0 == 0:
                    # means either 0-2 ratio if max_selected_1 is 1, else 0-1 ratio
                    ratio_positive_1_negative = \
                        group.loc[(group['selected0'] == max_selected_0) & (group['selected1'] ==
                                                                            negative_index)].iloc[0][
                            'count_normalized']
                    # max_selected_1 being 1 means 2 is the negative index: 1-2 ratio
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
                # Instead of generating the two possible alpha values for a comparison (i.e., treating selected0 as anchor versus
                # treating selected1 as anchor), let's extract only the dominant alpha value for each comparison (i.e., max difference).
                diff_positive_1_anchor = max_row['count_normalized'] - ratio_positive_1_negative
                diff_positive_2_anchor = max_row['count_normalized'] - ratio_positive_2_negative
                if ratio_positive_1_negative < ratio_positive_2_negative:
                    alpha_positive_1_positive_2 = diff_positive_1_anchor
                    alpha_positive_2_positive_1 = diff_positive_2_anchor
                else:
                    alpha_positive_1_positive_2 = diff_positive_1_anchor
                    alpha_positive_2_positive_1 = diff_positive_2_anchor
                # max_diff = max(diff_positive_1_anchor, diff_positive_2_anchor)
                # alpha_positive_1_positive_2 = max_row['count_normalized'] - ratio_positive_1_negative
                # alpha_positive_2_positive_1 = max_row['count_normalized'] - ratio_positive_2_negative
                if max_row['efforts_tuples'] == '[-1,-1,-1,0]_[0,-1,-1,1]':
                    print(f"ALPHAS: {alpha_positive_1_positive_2} . {alpha_positive_2_positive_1}")

                # Concatenate selected0 and selected1 to pattern match the alpha anchor_positive column
                alpha_selected_0_selected_1_column = f"alpha_{max_selected_0}_{max_selected_1}"

                alpha_selected_1_selected_0_column = f"alpha_{max_selected_1}_{max_selected_0}"

                df_comparisons.loc[max_row.name, alpha_selected_0_selected_1_column] = alpha_positive_1_positive_2
                df_comparisons.loc[max_row.name, alpha_selected_1_selected_0_column] = alpha_positive_2_positive_1
                comparisons_list.append(df_comparisons.loc[max_row.name])

            alpha_dataframes = pd.DataFrame(comparisons_list)
            alpha_dataframes.reset_index(drop=True, inplace=True)
            # print(f'{alpha_dataframes=}')
            # alpha_dataframes = alpha_dataframes[:10]
            # print("-------------------------------------")
            # print(f'{alpha_dataframes=}')
            return alpha_dataframes

        def _populate_alpha_matrices_and_masks(df_alphas):
            """
            Populate the alpha matrices and masks based on the data in the comparisons DataFrame.

            Args:
               None

           Returns:
               None
            """
            # Iterate over the rows of the comparisons DataFrame
            counter_df_alphas_rows = 0
            repeat_class_comparison_counter = 0
            equal_comparison_counter = 0
            zero_alphas_counter = 0
            unequal_comparison_counter = 0
            bool_swap_left_right = False
            # write out df_alphas to csv
            df_alphas.to_csv('py_df_alphas_walking.csv')
            for index, row in df_alphas.iterrows():
                counter_df_alphas_rows += 1
                # print(f"df_alphas_rows: {index}")
                efforts_tuple = row['efforts_tuples']
                # enforce constraint that i < j always corresponds to left, right / i > j to right, left effort_tuples.
                # where labels are efforts_tuple values (efforts_tuple[0] < efforts_tuple[1] based on R's
                # pmin, pmax functions) and i and j are indices to dict_similarity_classes_exemplars.keys()
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
                    print(f"repeat class comparison at indices: {index_left} , {index_right}")
                    continue
                # alpha_tripletid1_tripletid2 denotes one of the two alpha values per triplet as a function of the two most similar cases.
                # For any comparison, left_index < right_index
                # left, right indices indicate location for left, neut anchor_positive alpha with respect to Left,
                # Neutral Matrix (and right, neut anchor_positive alpha with respect to Right, Neutral Matrix)
                # whereas right, left indices indicate location for neut, left anchor_positive alpha and neut,
                # right anchor_positive alpha, respectively.
                if row['alpha_0_2'] != 0:
                    # print(f"entered alpha_0_2 with alphas: {row['alpha_0_2']} and {row['alpha_2_0']}")
                    # extract the two alpha values for the comparison, abiding by constraint
                    left_right_alpha = row['alpha_0_2']
                    right_left_alpha = row['alpha_2_0']
                    if bool_swap_left_right:
                        left_right_alpha = row['alpha_2_0']
                        right_left_alpha = row['alpha_0_2']
                    if left_right_alpha == 0 and right_left_alpha == 0:
                        zero_alphas_counter += 1
                        # print(f"left_right_alpha, right_left_alpha, both alphas zero...counter: {zero_alphas_counter}")
                        bool_constant_left_right = 0
                        bool_constant_right_left = 0
                    elif left_right_alpha == 0:
                        # print(f"left_right_alpha: {right_left_alpha}")
                        bool_constant_left_right = 0
                        bool_constant_right_left = 1
                        unequal_comparison_counter += 1
                        # print(f"unequal comparison counter: {unequal_comparison_counter}")
                    elif right_left_alpha == 0:
                        # print(f"right_left_alpha: {right_left_alpha}")
                        bool_constant_left_right = 1
                        bool_constant_right_left = 0
                        unequal_comparison_counter += 1
                        # print(f"unequal comparison counter: {unequal_comparison_counter}")
                    else:
                        # assert False, "left_right_alpha and right_left_alpha are both non-zero"
                        bool_constant_left_right = 1
                        bool_constant_right_left = 1
                        unequal_comparison_counter += 1
                        # print(f"unequal comparison counter: {unequal_comparison_counter}")

                    self.matrix_alpha_left_right_right_left.assign(tf.tensor_scatter_nd_add(
                        self.matrix_alpha_left_right_right_left,
                        indices=tf.constant([[index_left, index_right]]),
                        updates=tf.constant([left_right_alpha], dtype=tf.float32)
                    ))
                    self.matrix_bool_left_right.assign(tf.tensor_scatter_nd_update(
                        self.matrix_bool_left_right,
                        indices=tf.constant([[index_left, index_right]]),
                        updates=tf.constant([bool_constant_left_right], dtype=tf.float32)
                    ))
                    self.matrix_alpha_left_right_right_left.assign(tf.tensor_scatter_nd_add(
                        self.matrix_alpha_left_right_right_left,
                        indices=tf.constant([[index_right, index_left]]),
                        updates=tf.constant([right_left_alpha], dtype=tf.float32)
                    ))
                    self.matrix_bool_right_left.assign(tf.tensor_scatter_nd_update(
                        self.matrix_bool_right_left,
                        indices=tf.constant([[index_right, index_left]]),
                        updates=tf.constant([bool_constant_right_left], dtype=tf.float32)
                    ))
                elif row['alpha_0_1'] != 0:
                    # print(f"entered alpha_0_1 with alphas: {row['alpha_0_1']} and {row['alpha_1_0']}")
                    left_neutral_alpha = row['alpha_0_1']
                    neutral_left_alpha = row['alpha_1_0']
                    if bool_swap_left_right:
                        # left_neutral_alpha = row['alpha_2_1']
                        # neutral_left_alpha = row['alpha_1_2']
                        left_neutral_alpha = row['alpha_1_0']
                        neutral_left_alpha = row['alpha_0_1']
                    if left_neutral_alpha == 0 and neutral_left_alpha == 0:
                        zero_alphas_counter += 1
                        # print(f"left_neut, neut_left, both alphas zero...counter: {zero_alphas_counter}")
                        bool_constant_left_neutral = 0
                        bool_constant_neutral_left = 0
                    elif left_neutral_alpha == 0:
                        # print(f"left_neutral_alpha: {neutral_left_alpha}")
                        bool_constant_left_neutral = 0
                        bool_constant_neutral_left = 1
                        unequal_comparison_counter += 1
                        # print(f"unequal comparison counter: {unequal_comparison_counter}")
                    elif neutral_left_alpha == 0:
                        # print(f"neutral_left_alpha: {neutral_left_alpha}")
                        bool_constant_left_neutral = 1
                        bool_constant_neutral_left = 0
                        unequal_comparison_counter += 1
                        # print(f"unequal comparison counter: {unequal_comparison_counter}")
                    else:
                        # assert False, "left_neutral_alpha and neutral_left_alpha are both non-zero"
                        bool_constant_left_neutral = 1
                        bool_constant_neutral_left = 1
                        unequal_comparison_counter += 1
                        # print(f"unequal comparison counter: {unequal_comparison_counter}")
                    self.matrix_alpha_left_neut_neut_left.assign(tf.tensor_scatter_nd_add(
                        self.matrix_alpha_left_neut_neut_left,
                        indices=tf.constant([[index_left, index_right]]),
                        updates=tf.constant([left_neutral_alpha], dtype=tf.float32)
                    ))
                    self.matrix_bool_left_neut.assign(tf.tensor_scatter_nd_update(
                        self.matrix_bool_left_neut,
                        indices=tf.constant([[index_left, index_right]]),
                        updates=tf.constant([bool_constant_left_neutral], dtype=tf.float32)
                    ))
                    self.matrix_alpha_left_neut_neut_left.assign(tf.tensor_scatter_nd_add(
                        self.matrix_alpha_left_neut_neut_left,
                        indices=tf.constant([[index_right, index_left]]),
                        updates=tf.constant([neutral_left_alpha], dtype=tf.float32)
                    ))
                    self.matrix_bool_neut_left.assign(tf.tensor_scatter_nd_update(
                        self.matrix_bool_neut_left,
                        indices=tf.constant([[index_right, index_left]]),
                        updates=tf.constant([bool_constant_neutral_left], dtype=tf.float32)
                    ))
                elif row['alpha_2_1'] != 0:
                    # print(f"entered alpha_2_1 with alphas: {row['alpha_2_1']} and {row['alpha_1_2']}")
                    right_neutral_alpha = row['alpha_2_1']
                    neutral_right_alpha = row['alpha_1_2']
                    if bool_swap_left_right:
                        # right_neutral_alpha = row['alpha_0_1']
                        # neutral_right_alpha = row['alpha_1_0']
                        right_neutral_alpha = row['alpha_1_2']
                        neutral_right_alpha = row['alpha_2_1']
                    if right_neutral_alpha == 0 and neutral_right_alpha == 0:
                        zero_alphas_counter += 1
                        # print(f"right_neut, neut_right, both alphas zero...counter: {zero_alphas_counter}")
                        bool_constant_right_neutral = 0
                        bool_constant_neutral_right = 0
                    elif right_neutral_alpha == 0:
                        # print(f"right_neutral_alpha: {neutral_right_alpha}")
                        bool_constant_right_neutral = 0
                        bool_constant_neutral_right = 1
                        unequal_comparison_counter += 1
                        # print(f"unequal comparison counter: {unequal_comparison_counter}")
                    elif neutral_right_alpha == 0:
                        # print(f"neutral_right_alpha: {neutral_right_alpha}")
                        bool_constant_right_neutral = 1
                        bool_constant_neutral_right = 0
                        unequal_comparison_counter += 1
                        # print(f"unequal comparison counter: {unequal_comparison_counter}")
                    else:
                        # assert False, "right_neutral_alpha and neutral_right_alpha are both non-zero"
                        bool_constant_right_neutral = 1
                        bool_constant_neutral_right = 1
                        unequal_comparison_counter += 1
                        # print(f"unequal comparison counter: {unequal_comparison_counter}")
                    self.matrix_alpha_right_neut_neut_right.assign(tf.tensor_scatter_nd_add(
                        self.matrix_alpha_right_neut_neut_right,
                        indices=tf.constant([[index_left, index_right]]),
                        updates=tf.constant([right_neutral_alpha], dtype=tf.float32)
                    ))
                    self.matrix_bool_right_neut.assign(tf.tensor_scatter_nd_update(
                        self.matrix_bool_right_neut,
                        indices=tf.constant([[index_left, index_right]]),
                        updates=tf.constant([bool_constant_right_neutral], dtype=tf.float32)
                    ))
                    self.matrix_alpha_right_neut_neut_right.assign(tf.tensor_scatter_nd_add(
                        self.matrix_alpha_right_neut_neut_right,
                        indices=tf.constant([[index_right, index_left]]),
                        updates=tf.constant([neutral_right_alpha], dtype=tf.float32)
                    ))
                    self.matrix_bool_neut_right.assign(tf.tensor_scatter_nd_update(
                        self.matrix_bool_neut_right,
                        indices=tf.constant([[index_right, index_left]]),
                        updates=tf.constant([bool_constant_neutral_right], dtype=tf.float32)
                    ))
                # implies equal selection (or no selection) across all three pairs of a triplet
                else:
                    equal_comparison_counter += 1
                bool_swap_left_right = False
            print(f" Equal comparison counter: {equal_comparison_counter}")
            print(f"Unequal comparison counter: {unequal_comparison_counter}")

        # preprocess comparison data code execution starts here
        # read_in R generated similarity comparisons ratios csv
        aux_folder_path = (Path(__file__) / '../../aux').resolve()
        csv_similarity_ratios_path = aux_folder_path / f'{anim_name}_similarity_comparisons_ratios.csv'
        df_comparisons = pd.read_csv(csv_similarity_ratios_path)
        # Split the efforts_tuples values at the delimiter '_' and convert tokens to tuples
        df_comparisons['efforts_tuples'] = df_comparisons['efforts_tuples'].apply(
            lambda x: [tuple(ast.literal_eval(token)) for token in x.split('_')])
        set_comparison_classes = set([efforts_tuple[0] for efforts_tuple in df_comparisons['efforts_tuples']]).union(
            set([efforts_tuple[1] for efforts_tuple in df_comparisons['efforts_tuples']]))
        print(f"len set: {len(set_comparison_classes)}")

        # dict_similarity_classes_exemplars = {key: value for key, value in
        #                                      dict_similarity_classes_exemplars.items() if
        #                                      key in
        #                                      set_comparison_classes}

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
        _populate_alpha_matrices_and_masks(df_alphas)
        print(f"type of matrix_bool_left_right: {type(self.matrix_bool_left_right)}")
