# Based on: https://github.com/ringer-softwares/neuralnet
# from keras.callbacks import Callback
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_curve
import numpy as np

from .. import logger
from ..numpy import sigmoid


class SP(Callback):

    def __init__(self,
                 validation_data,
                 sample_weight=None,
                 verbose=False,
                 save_the_best=False,
                 patience=False,
                 from_logits=False):

        super().__init__()
        self.verbose = verbose
        self.patience = patience
        self.save_the_best = save_the_best
        self.sample_weight = sample_weight
        self.from_logits = from_logits

        self.count = 0
        self.__best_sp = -1.0
        self.__best_weights = None
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs={}):

        y_true = self.validation_data[1]
        y_hat = self.model.predict(
            self.validation_data[0], batch_size=1024).ravel()
        if self.from_logits:
            y_hat = sigmoid(y_hat)  # Apply sigmoid to logits
        # Computes SP
        fa, pd, thresholds = roc_curve(y_true, y_hat, sample_weight=self.sample_weight)
        sp = np.sqrt(np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa))))

        knee = np.argmax(sp)
        logs['val_max_sp'] = sp[knee]
        logs['val_max_sp_fa'] = fa[knee]
        logs['val_max_sp_pd'] = pd[knee]
        logs['val_max_sp_threshold'] = thresholds[knee]

        if self.verbose:
            logger.info("val_sp: {:.4f} (fa:{:.4f},pd:{:.4f}), patience: {}".format(sp[knee],
                                                                                    fa[knee], pd[knee], self.count))

        if self.__best_sp < 0:
            self.__best_sp = sp[knee]
            if self.save_the_best:
                logger.info('Saving the best configuration here...')
                self.__best_weights = self.model.get_weights()
                logs['val_max_sp_best_epoch'] = epoch
        elif round(sp[knee], 4) > round(self.__best_sp, 4):
            self.__best_sp = sp[knee]
            if self.save_the_best:
                logger.info('Saving the best configuration here...')
                self.__best_weights = self.model.get_weights()
                logs['val_max_sp_best_epoch'] = epoch
            self.count = 0
        else:
            self.count += 1

        if self.count > self.patience:
            logger.info('Stopping the Training by SP...')
            self.model.stop_training = True

    def on_train_end(self, logs={}):

        if self.save_the_best:
            logger.info('Reload the best configuration into the current model...')
            try:
                self.model.set_weights(self.__best_weights)
            except Exception:
                logger.fatal("Its not possible to set the weights. abort")
                raise