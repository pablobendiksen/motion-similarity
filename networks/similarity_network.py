from keras.callbacks import ModelCheckpoint

import conf
import networks.custom_losses as custom_losses
from keras.optimizers import Adam
from networks.utilities import Utilities
from keras import callbacks
from keras.models import Sequential
import tensorflow as tf

# tf.config.experimental_run_functions_eagerly(True)

from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense, Lambda, Dropout, GlobalAveragePooling2D, \
    Reshape, Conv1D, Multiply
import logging
import os

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, GlobalAveragePooling2D, Dense, Dropout, Multiply
from tensorflow.keras.models import Model

logging.basicConfig(level=logging.DEBUG,
                    filename=os.path.basename(__file__) + '.log',
                    format="{asctime} [{levelname:8}] {process} {thread} {module}: {message}",
                    style="{")
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)


# embedding network and training
class SimilarityNetwork(Utilities):
    """
    The SimilarityNetwork class defines our own triplet similarity network.

    This class inherits from the `Utilities` class and is used to build, compile, and train a similarity learning model.

    Args:
        train_loader: The data loader for the training dataset.
        validation_loader: The data loader for the validation dataset.
        test_loader: The data loader for the test dataset.
        checkpoint_root_dir: The directory where model checkpoints will be saved.

    Attributes:
        train_set: The training dataset loader.
        validation_set: The validation dataset loader.
        test_set: The test dataset loader.
        exemplar_dim: The dimensions of the exemplar data.
        checkpoint_dir: The directory for saving model checkpoints.
        embedding_size: The size of the embedding produced by the network.
        callbacks: List of Keras callbacks for training.
    """

    def __init__(self, train_loader, validation_loader, test_loader, checkpoint_root_dir, triplet_modules, architecture_variant):
        super().__init__()
        # train_generator's job is to randomly, and lazily, load batches from disk
        # SIM NETWORK CLASS: exemplar dim: (137, 88)
        self.train_set = train_loader
        self.validation_set = validation_loader
        self.test_set = test_loader
        self.exemplar_dim = train_loader.exemplar_dim
        print(f"SIM NETWORK CLASS: exemplar dim: {self.exemplar_dim}")
        self.triplet_modules = triplet_modules
        self.architecture_variant = architecture_variant
        self.checkpoint_dir = checkpoint_root_dir
        self.embedding_size = conf.embedding_size
        self.callbacks = [callbacks.EarlyStopping(monitor='batch_triplet_loss', patience=15, restore_best_weights=True, mode='min', verbose=1)]
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(self.checkpoint_dir,
                                  f"{self.architecture_variant}_similarity_model_weights_epoch_{{epoch:02d}}_val_loss_{{val_loss:.2f}}.h5"),
            save_weights_only=True,
            save_best_only=True,  # Save only when validation loss improves
            monitor="val_batch_triplet_loss",  # Ensure it tracks validation loss
            mode="min",  # Save when val_loss decreases
            verbose=1
        )
        # Save weights every 10 epochs
        self.callbacks.extend([checkpoint_callback])
        self.network = Sequential()
        self.build_model()
        self.compile_model()

    def build_model(self):
        input_shape = (self.exemplar_dim[0], self.exemplar_dim[1], 1)  # (132, 88, 1)
        self.network = Sequential()

        if self.architecture_variant == 0:
            # Output size: (132,88,32)
            self.network.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu',
                                    input_shape=input_shape))
            self.network.add(Dropout(0.2))
            self.network.add(BatchNormalization())
            # Output size: (66,44,32)
            self.network.add(MaxPool2D(2, 2))
            # Output size: (64,42,64)
            self.network.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
            self.network.add(Dropout(0.2))
            self.network.add(BatchNormalization())

            self.network.add(Conv2D(128, 3, strides=1, activation='relu'))
            self.network.add(BatchNormalization())

            # Output size: (32,21,64)
            self.network.add(MaxPool2D(2, 2))
            # Output size: (132 * 88 * 32) = 43008
            self.network.add(Flatten())
            # Output size: (32)
            self.network.add(Dense(self.embedding_size))
            self.network.add(Dropout(0.2))
            # Print the model summary
            self.network.summary()

        elif self.architecture_variant == 1:
            # Feature extraction with deeper convolution
            self.network.add(Conv2D(32, 3, strides=1, activation='relu', padding='same', input_shape=input_shape))
            self.network.add(BatchNormalization())

            self.network.add(Conv2D(64, 3, strides=1, activation='relu', padding='same'))
            self.network.add(BatchNormalization())

            self.network.add(Conv2D(128, 3, strides=1, activation='relu', padding='same', dilation_rate=2))
            self.network.add(BatchNormalization())

            # Downsampling with MaxPooling (instead of strided convolution)
            self.network.add(MaxPool2D(pool_size=(2, 2)))
            # Output shape: (66, 44, 128) - Spatial size halved

            # Further feature extraction
            self.network.add(Conv2D(256, 3, strides=1, activation='relu', padding='same'))
            self.network.add(BatchNormalization())

            self.network.add(Conv2D(256, 3, strides=1, activation='relu', padding='same'))
            self.network.add(BatchNormalization())

            # Another MaxPooling for further downsampling
            self.network.add(MaxPool2D(pool_size=(2, 2)))
            # Output shape: (33, 22, 256) - Spatial size halved again

            # Flatten instead of GlobalAveragePooling
            self.network.add(Flatten())

            # Bottleneck dense layer
            self.network.add(Dense(self.embedding_size))
            self.network.add(Dropout(0.2))  # Retaining dropout from the original model

            # Print the model summary
            self.network.summary()


    # def build_model(self):
    #     input_shape = (self.exemplar_dim[0], self.exemplar_dim[1], 1)  # (132, 88, 1)
    #     inputs = Input(shape=input_shape)
    #
    #     # Feature extraction
    #     x = Conv2D(32, 3, strides=1, activation='relu', padding='same')(inputs)
    #     x = BatchNormalization()(x)
    #     x = Conv2D(64, 3, strides=1, activation='relu', padding='same')(x)
    #     x = BatchNormalization()(x)
    #     x = Conv2D(128, 3, strides=1, activation='relu', padding='same', dilation_rate=2)(x)
    #     x = BatchNormalization()(x)
    #
    #     # Downsampling
    #     x = MaxPool2D(pool_size=(2, 2))(x)
    #
    #     # Further feature extraction
    #     x = Conv2D(256, 3, strides=1, activation='relu', padding='same')(x)
    #     x = BatchNormalization()(x)
    #     x = Conv2D(256, 3, strides=1, activation='relu', padding='same')(x)
    #     x = BatchNormalization()(x)
    #
    #     # Another downsampling
    #     x = MaxPool2D(pool_size=(2, 2))(x)
    #
    #     # **Attention mechanism**
    #     attention = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)  # Generate attention map
    #     x = Multiply()([x, attention])  # Element-wise multiplication of feature maps with attention weights
    #
    #     # Global Average Pooling
    #     x = GlobalAveragePooling2D()(x)
    #
    #     x = Flatten()(x)
    #
    #     # Bottleneck dense layer
    #     x = Dense(self.embedding_size)(x)
    #     x = Dropout(0.2)(x)
    #
    #     # Build model
    #     self.network = Model(inputs, x)
    #
    #     # Print the model summary
    #     self.network.summary()

    def compile_model(self):
        """
        Compile the neural network with a custom triplet loss function.

        Compiles the neural network using the Adam optimizer and a custom triplet loss function.
        Also sets up the network for training.

        Args:
            None

        Returns:
            None
        """
        opt = Adam(learning_rate=0.0001, beta_1=0.5)
        try:
            custom_loss_closure = custom_losses.create_batch_triplet_loss(self.triplet_modules)
            self.network.compile(optimizer=opt, loss=custom_loss_closure,
                                 metrics=[custom_loss_closure], run_eagerly=True)
        except RuntimeError:
            custom_loss_closure = custom_losses.create_batch_triplet_loss(self.triplet_modules)
            self.network.compile(optimizer=opt, loss=custom_loss_closure,
                                 metrics=[custom_loss_closure])
        finally:
            self.network.summary()

    def run_model_training(self):
        """
       Train the neural network on the training dataset.

       Trains the neural network on the training dataset and uses the validation dataset for monitoring.

       Args:
           None

       Returns:
           None
       """

        self.network.fit(self.train_set, validation_data=self.validation_set, epochs=conf.n_similarity_epochs,
                         steps_per_epoch=self.train_set.__len__(),
                         validation_steps=self.validation_set.__len__(), callbacks=self.callbacks)

        weights_path = os.path.join(self.checkpoint_dir, "similarity_model_weights.h5")
        self.network.save_weights(weights_path)
        print(f"Model weights saved to {weights_path}")

    def evaluate(self):
        """
        Evaluate the model on the test dataset.

        Evaluates the trained model on the test dataset attribute.

        Args:
            None

        Returns:
            None
        """
        self.network.evaluate(self.test_set, steps=self.test_set.__len__())
