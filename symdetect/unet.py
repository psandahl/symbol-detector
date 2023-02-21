import pathlib

import tensorflow.keras as keras
import tensorflow.keras.layers as layers


def convolution_block(input, num_filters: int):
    """
    Convolution block containing two convolutions.

    Parameters:
        input: The input to the block.
        num_filters: The number of filters used in the block.

    Returns:
        The output from the block.
    """
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(num_filters, (3, 3), padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x


def encoder_block(input, num_filters: int):
    """
    Encoder block, containing convolution and downsampling.

    Parameters:
        input: The input to the block.
        num_filters: The number of filters used in the block.

    Returns:
        Tuple (output before downsampling, output after downsampling).
    """
    x = convolution_block(input, num_filters)
    p = layers.MaxPool2D((2, 2))(x)

    return x, p


def decoder_block(input, features, num_filters: int):
    """
    Decoder block. Containing upsampling and convolution.

    Parameters:
        input: The input to upsample.
        features: The features from the contraction path.
        num_filters: The number of filters used in the block.

    Returns:
        The output from the block.
    """
    x = layers.UpSampling2D((2, 2))(input)
    x = layers.Concatenate()([x, features])
    x = convolution_block(x, num_filters)

    return x


def build_model(input_shape: tuple[int, int, int]):
    """
    Build the U-Net model.

    Parameters:
        input_shape: Input shape, tuple (height, width, channels).
    """
    inputs = layers.Input(input_shape)

    # Contraction/encoder path.
    f1, p1 = encoder_block(inputs, 64)
    f2, p2 = encoder_block(p1, 128)
    f3, p3 = encoder_block(p2, 256)
    f4, p4 = encoder_block(p3, 512)

    bridge = convolution_block(p4, 1024)

    # Expansion/decoder path.
    d1 = decoder_block(bridge, f4, 512)
    d2 = decoder_block(d1, f3, 256)
    d3 = decoder_block(d2, f2, 128)
    d4 = decoder_block(d3, f1, 64)

    outputs = layers.Conv2D(1, (1, 1), padding='same',
                            activation='sigmoid')(d4)

    return keras.Model(inputs, outputs)


def build_and_train(train_seq, validation_seq,
                    image_size: tuple[int, int],
                    model_path: pathlib.Path) -> None:
    """
    Build and train the U-Net model.

    Parameters:
        train_seq: The Sequence for the training data.
        validations_seq: The Sequence for the validation data.
        image_size: The image size (height, width).
        model_path: The path to where to write the model.
    """
    keras.backend.clear_session()

    model = build_model(image_size + (1,))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=[keras.metrics.BinaryAccuracy()])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=3),
        keras.callbacks.TensorBoard(log_dir='./logs'),
        keras.callbacks.ModelCheckpoint(model_path, save_best_only=True)
    ]

    model.fit(x=train_seq, validation_data=validation_seq,
              epochs=25, callbacks=callbacks)
