from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, \
    Dense, Flatten, Dropout, BatchNormalization, LSTM
import tensorflow as tf
from sklearn.model_selection import train_test_split


class Classifier:

    def __init__(self, random_state=None, epochs=500,
                 batch_size=32, input_shape=None, class_name=None, model=None, num_classes=2,
                 optimizer='Adam', learning_rate=0.001, loss='binary_crossentropy', num_outputs=2):

        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.class_name = class_name
        self.loss = loss
        self.learning_rate = learning_rate
        self.model = model
        self.num_outputs = num_outputs
        self.num_classes=num_classes
        try:
            self.optimizer = getattr(tf.keras.optimizers, optimizer)
        except:
            raise NotImplementedError('optimizer not implemented in keras')

    def flush(self):

        self.training = None
        self.model = None

    def build_model(self):

        pass

    def compile(self):

        opt = self.optimizer(learning_rate=self.learning_rate)

        if self.num_outputs == 1:
            self.model.compile(optimizer=opt, loss=self.loss, metrics=['accuracy'])
        else:
            losses = {self.class_name[0]: self.loss,
                      self.class_name[1]: self.loss}

            self.model.compile(optimizer=opt, loss=losses, loss_weights=[0.5, 0.5],
                               metrics=['accuracy'])

    def fit(self, X, y, sample_weight=None, output=False, verbose=1, X_test=None, y_test=None, random_state=0):

        if self.model is None:
            raise NotImplementedError('Model not created')

        y_1, y_2 = y
        if X_test is None:
            X_train, X_test, y_1_train, y_1_test, y_2_train, y_2_test = \
                train_test_split(X, y_1, y_2, test_size=0.3, random_state=random_state)
        else:
            X_train = X
            y_1_train = y_1
            y_2_train = y_2
            y_1_test, y_2_test = y_test

        # self.batch_size = len(X)
        self.training = self.model.fit(X_train, [y_1_train, y_2_train], verbose=verbose, batch_size=self.batch_size,
                             epochs=self.epochs, validation_data=(X_test, [y_1_test, y_2_test]))

        if output:
            return self.training, X_test, [y_1_test, y_2_test]

    def predict(self, X):

        return self.model.predict(X)

    def decision_function(self, X):

        return self.predict(X)

    def to_json(self):

        return self.model.to_json()

    def save_weights(self, filepath):

        self.model.save(filepath=filepath)

    def evaluate(self, x, y):

        self.model.evaluate(x, y)


class ConvToxinet(Classifier):

    def __init__(self, random_state=None, epochs=500,
                 batch_size=32, input_shape=None, class_name=None, model=None, num_classes=2,
                 optimizer='Adam', learning_rate=0.001, loss='binary_crossentropy', num_outputs=2):

        Classifier.__init__(self, random_state=random_state,
                            epochs=epochs,
                            input_shape=input_shape,
                            class_name=class_name,
                            model=model,
                            optimizer=optimizer,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            num_classes=num_classes,
                            loss=loss,
                            num_outputs=num_outputs)

    def build_model(self):

        inp = Input(shape=self.input_shape)
        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inp)
        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization(axis=-1)(x)

        if self.num_outputs == 1:
            outputs = Dense(1, activation='sigmoid', name=self.class_name)(x)
        else:
            outputs = []
            for out in range(self.num_outputs):
                outputs.append(Dense(1, activation='sigmoid', name=self.class_name[out])(x))

        self.model = Model(inp, outputs)


class ConvLSTMToxinet(Classifier):

    def __init__(self, random_state=None, epochs=500,
                 batch_size=32, input_shape=None, class_name=None, model=None, num_classes=2,
                 optimizer='Adam', learning_rate=0.001, loss='binary_crossentropy', num_outputs=2):

        Classifier.__init__(self, random_state=random_state,
                            epochs=epochs,
                            input_shape=input_shape,
                            class_name=class_name,
                            model=model,
                            optimizer=optimizer,
                            batch_size=batch_size,
                            num_classes=num_classes,
                            learning_rate=learning_rate,
                            loss=loss,
                            num_outputs=num_outputs)

    def build_model(self):

        inp = Input(shape=self.input_shape)
        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inp)
        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization(axis=-1)(x)
        x = LSTM(200, activation='relu', return_sequences=True)(x)
        x = LSTM(100, activation='relu')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization(axis=-1)(x)

        if self.num_outputs == 1:
            if self.num_classes == 2:
                outputs = Dense(1, activation='sigmoid', name=self.class_name)(x)
            else:
                outputs = Dense(self.num_classes, activation='softmax', name=self.class_name)(x)
        else:
            outputs = []
            for out in range(self.num_outputs):
                outputs.append(Dense(1, activation='sigmoid', name=self.class_name[out])(x))

        self.model = Model(inp, outputs)
