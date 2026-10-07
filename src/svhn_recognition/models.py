"""Model families adapted from the project owner's completed experiment."""


def build_model(name: str):
    """Create one of two dense or two convolutional Keras classifiers."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("Training requires: python -m pip install -e '.[train]'") from exc

    layers = tf.keras.layers
    if name == "ann_baseline":
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(32, 32)), layers.Flatten(),
            layers.Dense(64, activation="relu"), layers.Dense(32, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ])
        rate = .001
    elif name == "ann_deep":
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(32, 32)), layers.Flatten(),
            layers.Dense(256, activation="relu"), layers.Dense(128, activation="relu"),
            layers.Dropout(.2), layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"), layers.Dense(32, activation="relu"),
            layers.BatchNormalization(), layers.Dense(10, activation="softmax"),
        ])
        rate = .0005
    elif name == "cnn_baseline":
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(32, 32, 1)),
            layers.Conv2D(16, 3, padding="same"), layers.LeakyReLU(negative_slope=.1),
            layers.Conv2D(32, 3, padding="same"), layers.LeakyReLU(negative_slope=.1),
            layers.MaxPooling2D(2), layers.Flatten(), layers.Dense(32),
            layers.LeakyReLU(negative_slope=.1), layers.Dense(10, activation="softmax"),
        ])
        rate = .001
    elif name == "cnn_regularized":
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(32, 32, 1)),
            layers.Conv2D(16, 3, padding="same"), layers.LeakyReLU(negative_slope=.1),
            layers.Conv2D(32, 3, padding="same"), layers.LeakyReLU(negative_slope=.1),
            layers.MaxPooling2D(2), layers.BatchNormalization(),
            layers.Conv2D(32, 3, padding="same"), layers.LeakyReLU(negative_slope=.1),
            layers.Conv2D(64, 3, padding="same"), layers.LeakyReLU(negative_slope=.1),
            layers.MaxPooling2D(2), layers.BatchNormalization(), layers.Flatten(),
            layers.Dense(32), layers.LeakyReLU(negative_slope=.1), layers.Dropout(.5),
            layers.Dense(10, activation="softmax"),
        ])
        rate = .001
    else:
        raise ValueError(f"Unknown model {name!r}")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

