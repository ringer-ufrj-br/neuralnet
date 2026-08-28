# Based on: https://github.com/ringer-softwares/neuralnet
from tensorflow import keras
import tensorflow as tf


def auc(y_true, y_pred, num_thresholds=2000):
    import tensorflow as tf

    auc = tf.metrics.auc(y_true, y_pred, num_thresholds=num_thresholds)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return auc


def f1_score(y_true, y_pred):
    import tensorflow as tf

    f1 = tf.contrib.metrics.f1_score(y_true, y_pred)[1]
    keras.backend.get_session().run(tf.local_variables_initializer())
    return f1


class categorical_sp(keras.metrics.AUC):
    def __init__(self, *args, **kwargs):
        if ["multi_label" in kwargs]:
            del kwargs["multi_label"]
        super().__init__(multi_label=True, *args, **kwargs)
        tf.config.run_functions_eagerly(True)

    def result(self):
        fa: tf.Tensor = self.false_positives / (self.true_negatives + self.false_positives + keras.backend.epsilon())
        pd: tf.Tensor = self.true_positives / (self.true_positives + self.false_negatives + keras.backend.epsilon())
        sp: tf.Tensor = tf.norm(
            keras.backend.sqrt(keras.backend.sqrt(pd * (1 - fa)) * (0.5 * (pd + (1 - fa)))),
            axis=1,
        )
        knee = keras.backend.argmax(sp)
        return sp[knee] / keras.backend.sqrt(keras.backend.variable(value=self._num_labels))


class sp(keras.metrics.AUC):
    # This implementation works with Tensorflow backend tensors.
    # That way, calculations happen faster and results can be seen
    # while training, not only after each epoch
    def result(self):

        # Add keras.backend.epsilon() for forbiding division by zero
        fa = self.false_positives / (self.true_negatives + self.false_positives + keras.backend.epsilon())
        pd = self.true_positives / (self.true_positives + self.false_negatives + keras.backend.epsilon())

        sp = keras.backend.sqrt(keras.backend.sqrt(pd * (1 - fa)) * (0.5 * (pd + (1 - fa))))
        knee = keras.backend.argmax(sp)
        return sp[knee]


class pd(keras.metrics.AUC):
    # This implementation works with Tensorflow backend tensors.
    # That way, calculations happen faster and results can be seen
    # while training, not only after each epoch
    def result(self):

        # Add keras.backend.epsilon() for forbiding division by zero
        fa = self.false_positives / (self.true_negatives + self.false_positives + keras.backend.epsilon())
        pd = self.true_positives / (self.true_positives + self.false_negatives + keras.backend.epsilon())

        sp = keras.backend.sqrt(keras.backend.sqrt(pd * (1 - fa)) * (0.5 * (pd + (1 - fa))))
        knee = keras.backend.argmax(sp)
        return pd[knee]


class fa(keras.metrics.AUC):
    # This implementation works with Tensorflow backend tensors.
    # That way, calculations happen faster and results can be seen
    # while training, not only after each epoch
    def result(self):

        # Add keras.backend.epsilon() for forbiding division by zero
        fa = self.false_positives / (self.true_negatives + self.false_positives + keras.backend.epsilon())
        pd = self.true_positives / (self.true_positives + self.false_negatives + keras.backend.epsilon())

        sp = keras.backend.sqrt(keras.backend.sqrt(pd * (1 - fa)) * (0.5 * (pd + (1 - fa))))
        knee = keras.backend.argmax(sp)
        return fa[knee]


categorical_sp_metric = categorical_sp(
    num_thresholds=1000,
    curve="ROC",
    summation_method="interpolation",
    name=None,
    dtype=None,
    thresholds=None,
    multi_label=True,
    label_weights=None,
)

sp_metric = sp(
    num_thresholds=1000,
    curve="ROC",
    summation_method="interpolation",
    name=None,
    dtype=None,
    thresholds=None,
    multi_label=False,
    label_weights=None,
)

pd_metric = pd(
    num_thresholds=1000,
    curve="ROC",
    summation_method="interpolation",
    name=None,
    dtype=None,
    thresholds=None,
    multi_label=False,
    label_weights=None,
)

fa_metric = fa(
    num_thresholds=1000,
    curve="ROC",
    summation_method="interpolation",
    name=None,
    dtype=None,
    thresholds=None,
    multi_label=False,
    label_weights=None,
)
