import math

import tensorflow as tf
import numpy as np
import pickle


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
    square_norm = tf.diag_part(dot_product)

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
    np.random.seed(42)
    tf.random.set_seed(42)
    # Generate random input tensors
    batch_size = 32
    embed_dim = 4

    embeddings = tf.constant(np.random.normal(size=(batch_size, embed_dim)), dtype=tf.float32)
    mask = tf.constant(np.random.uniform(size=(batch_size, batch_size)), dtype=tf.float32)

    embeddings = tf.random.normal(shape=(batch_size, embed_dim))
    mask = tf.random.uniform(shape=(batch_size, batch_size))

    # # Compute pairwise distances
    # pairwise_dist = _pairwise_distances(embeddings, mask)
    #
    # # Print the result
    # print(pairwise_dist)

    check_ = pickle.load(open("/Users/bendiksen/Desktop/research/vr_lab/motion-similarity-project/exemplars_dir/tmp" \
                          "//similarity_labels_exemplars_dict.pickle", "rb"))
    print(len(check_))
    print([len(value) for key, value in check_.items()])
