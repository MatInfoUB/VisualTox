from keras.models import Sequential, Input, Model
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization
import keras.optimizers
from sklearn.model_selection import train_test_split


class Classifier:

    def __init__(self, random_state=None, epochs=500,
                 batch_size=32, input_shape=None, class_name=None,
                 optimizer=None, learning_rate=0.001):

        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.class_name = class_name

        if optimizer is None:
            self.optimizer = 'Adam'
        else:
            self.optimizer = optimizer

        self.learning_rate = learning_rate

        self.build_model()

    def build_model(self, num_output=2):

        inp = Input(shape=self.input_shape)
        x = Conv1D(filters=32, kernel_size=3, activation='relu')(inp)
        # x = Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer='l2')(x)
        x = Dropout(0.5)(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
        # x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
        # x = Conv1D(filters=32, kernel_size=3, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = Dropout(0.3)(x)
        x = Dense(256)(x)
        x = Dropout(0.3)(x)
        x = Dense(128)(x)
        x = Dropout(0.3)(x)

        if num_output == 1:
            outputs = Dense(2, activation='softmax', name=self.class_name[1])(x)
        else:
            outputs = []
            for out in range(num_output):
                outputs.append(Dense(2, activation='softmax', name=self.class_name[out])(x))
            # output1 = Dense(2, activation='softmax', name=self.class_name[0])(x)
            # output2 = Dense(2, activation='softmax', name=self.class_name[1])(x)

        model = Model(inp, outputs)

        try:
            self.optimizer = getattr(keras.optimizers, self.optimizer)
        except:
            raise NotImplementedError('optimizer not implemented in keras')
        opt = self.optimizer(lr=self.learning_rate)
        losses = {self.class_name[0]: 'categorical_crossentropy',
                  self.class_name[1]: 'categorical_crossentropy'}

        model.compile(optimizer=opt, loss=losses, loss_weights=[0.5, 0.5],
                      metrics=['accuracy'])

        self.model = model

    def fit(self, X, y, sample_weight=None, output=False, verbose=1, X_test=None, y_test=None):

        y_1, y_2 = y
        if X_test is None:
            X_train, X_test, y_1_train, y_1_test, y_2_train, y_2_test = train_test_split(X, y_1, y_2, test_size=0.3)
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




