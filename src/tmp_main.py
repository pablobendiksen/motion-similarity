import math

import tensorflow as tf
import numpy as np

def _pairwise_distances(embeddings, mask, squared=False):
    """Compute the 2D matrix of distances between valid embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        mask: tensor of shape (batch_size, batch_size) indicating valid pairs
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    mask_expanded = tf.expand_dims(mask, axis=-1)  # Expand mask along the second dimension
    embeddings_expanded = tf.expand_dims(embeddings, axis=1)
    masked_embeddings = embeddings_expanded * mask_expanded  # Element-wise multiplication
    dot_product = tf.matmul(masked_embeddings, tf.transpose(embeddings))
    square_norm = tf.linalg.diag_part(dot_product)

    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
    distances = tf.maximum(distances, 0.0)

    if not squared:
        mask_float = tf.cast(mask, tf.float32)
        distances = distances * mask_float

        distances = distances + mask_float * 1e-16
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - mask_float)

    return distances

def _pairwise_distances_2(embeddings, mask, squared=False):
    mask = tf.cast(mask > 0.5, tf.float32)  # Convert the mask to binary (0 or 1)
    mask_sparse = tf.sparse.from_dense(mask)
    masked_embeddings = tf.sparse.sparse_dense_matmul(mask_sparse, embeddings, adjoint_b=True)
    masked_embeddings_dense = tf.sparse.to_dense(masked_embeddings)
    dot_product = tf.matmul(masked_embeddings_dense, tf.transpose(embeddings))  # Transpose the `embeddings` tensor
    square_norm = tf.linalg.diag_part(dot_product)

    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
    distances = tf.maximum(distances, 0.0)

    if not squared:
        mask_float = tf.cast(mask, tf.float32)
        distances = distances * mask_float

        distances = distances + mask_float * 1e-16
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - mask_float)

    return distances

if __name__ == '__main__':

    # np.random.seed(42)
    # tf.random.set_seed(42)
    # # Generate random input tensors
    # batch_size = 32
    # embed_dim = 4
    #
    # embeddings = tf.constant(np.random.normal(size=(batch_size, embed_dim)), dtype=tf.float32)
    # mask = tf.constant(np.random.uniform(size=(batch_size, batch_size)), dtype=tf.float32)
    #
    # # embeddings = tf.random.normal(shape=(batch_size, embed_dim))
    # # mask = tf.random.uniform(shape=(batch_size, batch_size))
    #
    # # Compute pairwise distances
    # pairwise_dist = _pairwise_distances_2(embeddings, mask)
    #
    # # Print the result
    # print(pairwise_dist)

    import itertools
    from collections import deque


    def exhaust(generator):
        generator = list(generator)
        print([x for x in generator])

    # exhaust(itertools.combinations(range(4), 2))
    elements = (-1, 0, 1)
    two_tuples = []
    three_tuples = []


    def generate_states_and_drives():

        def convertToNary(tuple_index, values_per_effort, efforts_per_tuple):
            effort_tuple = []
            zeroes_counter = 0
            for _ in range(efforts_per_tuple):
                tuple_index, remainder = divmod(tuple_index, values_per_effort)
                if remainder == 1:
                    zeroes_counter+=1
                effort_tuple.append(effort_vals[remainder])
            if zeroes_counter == 2 or zeroes_counter == 1:
                states_and_drives.append(effort_tuple)

        states_and_drives = []
        effort_vals = [-1,0,1]
        values_per_effort = 3
        efforts_per_tuple = 4
        tuple_index = 0
        while (tuple_index < math.pow(values_per_effort, 4)):
            tuple_index += 1
            convertToNary(tuple_index, values_per_effort, efforts_per_tuple)
        print(f"states_and_drives len: {len(states_and_drives)} \n{states_and_drives}")


    for i in range(10, 35+1, 3):
        indices = range(i - 10, i)
        print(indices)


    # # Remove duplicates and sort the tuples
    # tuples_ = sorted(set(tuples))
    #
    # # Print the generated tuples
    # cnt = 0
    # for tpl in tuples:
    #     cnt+=1
    #     print(tpl)
    # print(cnt)
