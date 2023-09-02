import ast
import pickle
import conf
import numpy as np
import pandas as pd
import tensorflow as tf

# Set the display options to show all columns
pd.set_option('display.max_columns', None)

# consider alternative loss function based on cosine similarity:
# self.margin = 1
# self.loss = tf.keras.losses.CosineSimilarity(axis=1)
# ap_distance = self.loss(anchor, positive)
# an_distance = self.loss(anchor, negative)
# loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)

#  user-defined loss function that computes a triplet loss for a batch of embeddings
def batch_all_triplet_loss(y_true, y_pred):
    """Build triplet loss over a batch of embeddings.

        We calculate triplet losses for all anchor-positive possibilities, and mask for semi-hard cases only.

        Args:
            y_true: supposed 'labels' of the batch (i.e., class indexes), tensor of size (batch_size,)
            y_pred: embeddings, tensor of shape (batch_size, embed_dim)

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
    pairs_embeddings_idxs = list(zip(y_pred, y_true))
    # sort the pairs based on the indexes
    pairs_embeddings_idxs.sort(key=lambda x: x[1])
    # separate the embeddings from the indexes
    y_pred, y_true = zip(*pairs_embeddings_idxs)
    triplet_mining = TripletMining()
    classes_distances = triplet_mining.calculate_left_right_distances(y_pred)
    triplet_mining.calculate_class_neut_distances()

    # case 1: left and right are positives
    # L = anchor
    triplet_loss_1 = (classes_distances - triplet_mining.tensor_dists_class_neut +
                      triplet_mining.matrix_alpha_left_right)
    triplet_loss_L_R = tf.multiply(triplet_loss_1, triplet_mining.matrix_bool_left_right)
    # R = anchor
    triplet_loss_R_L = tf.multiply(triplet_loss_1, triplet_mining.matrix_bool_right_left)

    # case 2: left and neutral are positives
    # L = anchor
    # Reshape the 1D tensor into a 2D tensor of shape (56, 1) (will allow for broadcasting)
    column_dists_class_neut = tf.reshape(triplet_mining.tensor_dists_class_neut, (triplet_mining.num_states_drives, 1))
    triplet_loss_2 = (column_dists_class_neut - classes_distances +
                      triplet_mining.matrix_alpha_left_neut)
    triplet_loss_L_N = tf.multiply(triplet_loss_2, triplet_mining.matrix_bool_left_neut)
    # N = anchor
    # Reshape the 1D tensor into a 2D tensor with shape (1, 56)
    row_dists_class_neut = tf.reshape(triplet_mining.tensor_dists_class_neut, (1, triplet_mining.num_states_drives))
    triplet_loss_2 = row_dists_class_neut - column_dists_class_neut + triplet_mining.matrix_alpha_left_neut
    triplet_loss_N_L = tf.multiply(triplet_loss_2, triplet_mining.matrix_bool_neut_left)

    # case 3: right and neutral are positives
    # R = anchor
    triplet_loss_3 = row_dists_class_neut - tf.transpose(classes_distances) + triplet_mining.matrix_alpha_right_neut
    triplet_loss_R_N = tf.multiply(triplet_loss_3, triplet_mining.matrix_bool_right_neut)
    # N = anchor
    triplet_loss_3 = column_dists_class_neut - row_dists_class_neut + triplet_mining.matrix_alpha_right_neut
    triplet_loss_N_R = tf.multiply(triplet_loss_3, triplet_mining.matrix_bool_neut_right)

    losses = (triplet_loss_L_R + triplet_loss_R_L + triplet_loss_L_N + triplet_loss_N_L + triplet_loss_R_N +
              triplet_loss_N_R)

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = tf.maximum(losses, 0.0)

    # Count number of positive err triplets (where triplet_loss > 0)
    valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), float)
    num_positive_triplets = tf.reduce_sum(valid_triplets)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss


class TripletMining:

    def __init__(self):
        self.dict_similarity_classes_exemplars = pickle.load(open(
            conf.exemplars_dir + conf.similarity_dict_file_name, "rb"))
        print(f"classes: {self.dict_similarity_classes_exemplars.keys()}")
        self.dict_neutral_exemplar = self.dict_similarity_classes_exemplars.pop((0, 0, 0, 0))
        self.num_states_drives = len(self.dict_similarity_classes_exemplars.keys())
        # assert self.num_states_drives == 56, f"Incorrect number of states + drives: {self.num_states_drives}"
        self.matrix_alpha_left_right, self.matrix_alpha_left_neut, self.matrix_alpha_right_neut = (np.zeros((
            self.num_states_drives, self.num_states_drives)) for _ in range(3))
        # Generate boolean matrices for each anchor_positive permutation.
        (self.matrix_bool_left_right, self.matrix_bool_right_left, self.matrix_bool_left_neut,
         self.matrix_bool_neut_left, self.matrix_bool_right_neut, self.matrix_bool_neut_right) \
            = (np.full((self.num_states_drives, self.num_states_drives), 0) for _ in range(6))
        print(self.matrix_alpha_left_right.shape)
        print(self.matrix_bool_right_left.shape)
        self.tensor_dists_left_right_right_left = tf.Variable(
            initial_value=tf.zeros((self.num_states_drives, self.num_states_drives)))
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
        if not conf.bool_fixed_neutral_embedding:
            embeddings = self._extract_neutral_embedding(embeddings)
        # shape (batch_size, batch_size)
        dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

        # Get squared L2 norm for each embedding (each embedding's dot product with itself
        # shape (batch_size,)
        square_norm = tf.linalg.diag_part(dot_product)

        # Compute the pairwise distance matrix as we have:
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
            tf.debugging.assert_shapes([(distances, (tf.TensorShape([conf.similarity_batch_size - 1,
                                                                     conf.similarity_batch_size - 1]),))])
        else:
            tf.debugging.assert_shapes([(distances, (tf.TensorShape([conf.similarity_batch_size,
                                                                     conf.similarity_batch_size]),))])
        return distances

    def calculate_class_neut_distances(self, embeddings):
        """Calculate 1D tensor of L2 norm of differences between class embeddings and the neutral embedding of
                shape (batch_size,)

                Args:
                    embeddings: tensor of shape (batch_size, embed_dim)

                Returns:
                    None
        """
        self.tensor_dists_class_neut.assign(tf.norm(embeddings - self.neutral_embedding, axis=1))

    def batch_semi_hard_triplet_loss(self, y_true, y_pred):
        pass

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
            print(f'{df_alphas=}')
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
                    self.matrix_alpha_left_right[index_left, index_right] = left_right_alpha
                    self.matrix_bool_left_right[index_left, index_right] = 1
                    self.matrix_alpha_left_right[index_right, index_left] = right_left_alpha
                    self.matrix_bool_right_left[index_right, index_left] = 1
                elif row['alpha_0_1'] != 0:
                    print("entered alpha_0_1")
                    left_neutral_alpha = row['alpha_0_1']
                    neutral_left_alpha = row['alpha_1_0']
                    if bool_swap_left_right:
                        left_neutral_alpha = row['alpha_2_1']
                        neutral_left_alpha = row['alpha_1_2']
                    self.matrix_alpha_left_neut[index_left, index_right] = left_neutral_alpha
                    self.matrix_bool_left_neut[index_left, index_right] = 1
                    self.matrix_alpha_left_neut[index_right, index_left] = neutral_left_alpha
                    self.matrix_bool_neut_left[index_right, index_left] = 1
                elif row['alpha_2_1'] != 0:
                    print(f"entered alpha_2_1 at indices: {index_left} , {index_right}")
                    right_neutral_alpha = row['alpha_2_1']
                    neutral_right_alpha = row['alpha_1_2']
                    if bool_swap_left_right:
                        right_neutral_alpha = row['alpha_0_1']
                        neutral_right_alpha = row['alpha_1_0']
                    self.matrix_alpha_right_neut[index_left, index_right] = right_neutral_alpha
                    print(f"{self.matrix_alpha_right_neut[index_left, index_right]}")
                    self.matrix_bool_right_neut[index_left, index_right] = 1
                    self.matrix_alpha_right_neut[index_right, index_left] = neutral_right_alpha
                    self.matrix_bool_neut_right[index_right, index_left] = 1
                # implies equal selection (or no selection) across all three pairs of a triplet
                else:
                    equal_comparison_counter += 1
                bool_swap_left_right = False

            print(f'{self.matrix_alpha_right_neut[16]}')
            print(f'{len(self.matrix_alpha_right_neut[16])}')
            print(f'{self.matrix_bool_right_neut[16]}')
            print(f'{len(self.matrix_bool_right_neut[16])}')
            print(f'{repeat_class_comparison_counter=}')
            print(f'{equal_comparison_counter=}')

        def _convert_np_matrices_to_tensors():
            matrices_list = [self.matrix_alpha_left_right, self.matrix_alpha_left_neut, self.matrix_alpha_right_neut]
            (self.matrix_alpha_left_right, self.matrix_alpha_left_neut, self.matrix_alpha_right_neut) \
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


#
if __name__ == '__main__':
    my_tensor = tf.Variable(initial_value=tf.constant([[2, 3, 4], [6, 7, 8]]))
    my_tensor = my_tensor.assign(tf.zeros((2, 3), dtype=np.int32))
    diag = tf.cast(tf.equal(my_tensor, 0), float)
    tf.print(diag)
    triplet_mining_2 = TripletMining()
    # possible eval metrics: counting the triplets for which the positive distance (anchor - positive) is less than
    # the negative distance (anchor - negative) (by at least the margin) and then dividing by the total number of
    # triplets in the batch (i.e., proportion of zero loss triplets)

    # classification: model embeddings to classify motions (take an unseen motion exemplar (representing one class) and
    # compare it - using L2 normalized Euclidean distance - with its nearest neighbor.

    # ranking accuracy:
    # compute L2 normalized distances between all pairs of classes in the embedding space, and generate Spearman
    # rank correlation coefficient (SROCC) between the sorted distances (ascending order) and the complement of the
    # user normalized similarity scores (i.e., 1 - normalized similarity scores).

