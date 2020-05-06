import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20
import pandas as pd

import matplotlib.cm as cm

from rdkit import Chem
from rdkit.Chem import Draw

opts = Draw.DrawingOptions()
opts.atomLabelFontSize = 110
opts.dotsPerAngstrom = 200
opts.bondLineWidth = 6.0

from toxinet import load_training_data, create_new_predicted_data, \
    ConvLSTMToxinet, Activation, EntropySelection, load_evaluation_data, balance_data
pred_figdir = os.path.join(os.getcwd(), 'figdir', 'prediction')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

X, y, class_name, new_data = load_training_data(balanced=True, augment=True)

lbe = LabelEncoder().fit(y[0])
y_1 = lbe.transform(y[0])
y_2 = lbe.transform(y[1])


X_val, y_val, val_data = load_evaluation_data(class_name, balanced=True, augment=True)

y_1_val = lbe.transform(y_val[0])
y_2_val = lbe.transform(y_val[1])
input_shape = X.shape[1:]

total_epochs = 3000
max_iter = 20
k = 100


epochs_per_k = int(total_epochs / max_iter)
from sklearn.metrics import roc_auc_score, average_precision_score

auc = []

Agonist_Class_acc = []
Binding_Class_acc = []
val_Agonist_Class_acc = []
val_Binding_Class_acc = []


X_train, X_test, y_1_train, y_1_test, y_2_train, y_2_test = \
    train_test_split(X, y_1, y_2, test_size=0.3, random_state=0)

model = ConvLSTMToxinet(epochs=epochs_per_k, input_shape=input_shape,
                       class_name=class_name, batch_size=2000)
model.build_model()
model.compile()
auc_1 = []
auc_2 = []
pr_1 = []
pr_2 = []

ag_1 = []
bin_1 = []
ag_2 = []
bin_2 = []

for i in range(max_iter):
    print('Active Learning for Iteration: ', i)

    # training = model.model.fit(X[train], [y_1[train], y_2[train]],
    #                    verbose=0, batch_size=len(train), epochs=epochs_per_k,
    #                    validation_data=(X[test], [y_1[test], y_2[test]]))

    training = model.model.fit(X_train, [y_1_train, y_2_train],
                               verbose=0, batch_size=len(X_train), epochs=epochs_per_k,
                               validation_data=(X_test, [y_1_test, y_2_test]))

    print('Training Completed. Calculating accuracy values')
    print('Agonist Class accuracy is: ', training.history['val_Agonist_Class_acc'][-1])
    print('Binding Class accuracy is: ', training.history['val_Binding_Class_acc'][-1])

    ag_1.append(training.history['Agonist_Class_acc'])
    bin_1.append(training.history['Binding_Class_acc'])
    ag_2.append(training.history['val_Agonist_Class_acc'])
    bin_2.append(training.history['val_Binding_Class_acc'])

    y_score = model.model.predict(X_test)
    auc_1.append(roc_auc_score(y_1_test, y_score[0]))
    auc_2.append(roc_auc_score(y_2_test, y_score[1]))
    pr_1.append(average_precision_score(y_1_test, y_score[0]))
    pr_2.append(average_precision_score(y_2_test, y_score[1]))

    print('Calculating Uncertainties of Second Data')
    probs = model.predict(X_val)
    probs = 0.5 * (probs[0] + probs[1])

    k_vals = EntropySelection.select(probs, k)

    uncertain_samples = np.zeros(len(X_val), dtype='bool')
    uncertain_samples[k_vals] = True  # Index of uncertain samples from X_val

    X_train = np.concatenate((X_train, X_val[uncertain_samples]))
    y_1_train = np.concatenate((y_1_train, y_1_val[uncertain_samples]))
    y_2_train = np.concatenate((y_2_train, y_2_val[uncertain_samples]))

    print(sum(uncertain_samples), ' Data points have been sampled')
    print('Total size of Training + Testing data: ', len(X))

    X_val = X_val[~uncertain_samples]
    y_1_val = y_1_val[~uncertain_samples]
    y_2_val = y_2_val[~uncertain_samples]
    val_data = val_data[~uncertain_samples]

    print('Remaining Validation Dataset size: ', len(X_val))

scores = model.model.evaluate(X_test, [y_1_test, y_2_test])
print(scores)

ag_1 = np.asarray(ag_1).flatten()
bin_1 = np.asarray(bin_1).flatten()
ag_2 = np.asarray(ag_2).flatten()
bin_2 = np.asarray(bin_2).flatten()

Agonist_Class_acc.append(ag_1)
val_Agonist_Class_acc.append(ag_2)
Binding_Class_acc.append(bin_1)
val_Binding_Class_acc.append(bin_2)

model.model.save('results/al_model_lstm.h5')

import pandas as pd

acc_names = ['Agonist (Training)', 'Agonist (Validation)',
             'Binding (Training)', 'Binding (Validation)']


accuracies = pd.DataFrame({acc_names[0]: ag_1, acc_names[1]: bin_1,
                           acc_names[2]: ag_2, acc_names[3]: bin_2})

accuracies.to_csv('AL_training.csv')