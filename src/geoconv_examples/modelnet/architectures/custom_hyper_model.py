import keras_tuner as kt
import tensorflow as tf


class CustomHyperModel(kt.HyperModel):
    def __init__(self, build_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.build_fn = build_fn

    def build(self, hp):
        return self.build_fn(hp)

    def fit(self, hp, model, *args, **kwargs):
        training_history = model.fit(*args, **kwargs)
        tf.keras.backend.clear_session(free_memory=True)
        return training_history
