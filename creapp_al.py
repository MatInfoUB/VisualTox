import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20

import matplotlib.cm as cm

from rdkit import Chem
from rdkit.Chem import Draw

opts = Draw.DrawingOptions()
opts.atomLabelFontSize = 110
opts.dotsPerAngstrom = 200
opts.bondLineWidth = 6.0

from toxinet import load_training_data, create_new_predicted_data, \
    Classifier, Activation, EntropySelection, load_evaluation_data
figdir = os.path.join(os.getcwd(), 'creapp_figs')

X, y, class_name, new_data = load_training_data(balanced=True)
from sklearn.model_selection import train_test_split
y_1, y_2 = y


X_val, y_val, _ = load_evaluation_data(class_name)
y_1_val, y_2_val = y_val
input_shape = X.shape[1:]

max_iter = 20
k = 200

from sklearn.metrics import roc_auc_score

auc = []

for i in range(max_iter):

    X_train, X_test, y_1_train, y_1_test, y_2_train, y_2_test = train_test_split(X, y_1, y_2, test_size=0.3)
    model = Classifier(epochs=1000, input_shape=input_shape, class_name=class_name)
    training = model.model.fit(X_train, [y_1_train, y_2_train], verbose=1, batch_size=len(X_train),
                               epochs=1000, validation_data=(X_test, [y_1_test, y_2_test]))

    y_score = model.model.predict(X_test)
    auc.append(roc_auc_score(y_1_test, y_score[0]))

    probs = model.model.predict(X_val)
    probs = 0.5 * (probs[0] + probs[1])

    uncertain_samples = np.zeros(len(X_val), dtype='bool')
    uncertain_samples[EntropySelection.select(probs, k)] = True

    X = np.concatenate((X, X_val[uncertain_samples]))
    y_1 = np.concatenate((y_1, y_1_val.iloc[uncertain_samples]))
    y_2 = np.concatenate((y_2, y_2_val.iloc[uncertain_samples]))

    X_val = X_val[~uncertain_samples]
    y_1_val = y_1_val.iloc[~uncertain_samples]
    y_2_val = y_2_val.iloc[~uncertain_samples]
