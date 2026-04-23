"""
Brain MRI Abnormality Detection - Autoencoder Model
Author: Manoj Kumar Sao | Enrollment: 2452448072 | MCA 4th Sem | IGNOU
"""

import os

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Input, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_autoencoder(input_shape=(128, 128, 1)):
    """
    Convolutional autoencoder for unsupervised Brain MRI anomaly detection.
    It learns normal MRI structure and uses reconstruction error to highlight abnormalities.
    """
    inputs = Input(shape=input_shape, name="mri_input")

    x = Conv2D(32, (3, 3), padding="same", name="enc_conv1")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), name="enc_pool1")(x)

    x = Conv2D(64, (3, 3), padding="same", name="enc_conv2")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2), name="enc_pool2")(x)

    x = Conv2D(128, (3, 3), padding="same", name="enc_conv3")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    encoded = MaxPooling2D((2, 2), name="enc_pool3")(x)

    x = UpSampling2D((2, 2), name="dec_up1")(encoded)
    x = Conv2D(128, (3, 3), padding="same", name="dec_conv1")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D((2, 2), name="dec_up2")(x)
    x = Conv2D(64, (3, 3), padding="same", name="dec_conv2")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = UpSampling2D((2, 2), name="dec_up3")(x)
    x = Conv2D(32, (3, 3), padding="same", name="dec_conv3")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    decoded = Conv2D(1, (1, 1), activation="sigmoid", name="output")(x)

    autoencoder = Model(inputs, decoded, name="BrainMRI_Autoencoder")
    autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return autoencoder


def get_callbacks(model_path, monitor="val_loss"):
    return [
        ModelCheckpoint(model_path, save_best_only=True, monitor=monitor, verbose=1),
        EarlyStopping(patience=10, restore_best_weights=True, monitor=monitor),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6, monitor=monitor, verbose=1),
    ]


def train_autoencoder(model, X_train, X_val, epochs=50, batch_size=32, save_path="models/autoencoder.h5"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    history = model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=get_callbacks(save_path),
        verbose=1,
    )
    return history


if __name__ == "__main__":
    ae = build_autoencoder()
    ae.summary()
