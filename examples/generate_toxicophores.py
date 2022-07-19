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

from toxinet.load_data import *
from toxinet import Activation, Classifier
from tensorflow.keras.models import load_model

model = Classifier(model=load_model('results/al_model_lstm.h5'))
act = Activation(model)

actives = pd.read_csv('data/0422_all_edc_inactives.csv')[['Name', 'Canonical']]
actives['Smile_length'] = actives['Canonical'].apply(lambda x: len(x))

smi = Smile()
X = smi.smiles_to_sequences(actives['Canonical'])

maxlen = 130

X = sequence.pad_sequences(X, maxlen=maxlen)
X = X.astype(np.float32) / (np.float32(smi.max_num))
X = X.reshape(len(X), maxlen, 1)

start_ind = 0
end_ind = 100

dataset = actives
import time

t = time.time()
y_test = model.model.predict(X)
print('Total Prediction Time :', time.time() - t, ' seconds')

class_names = ['Agonist Class', 'Binding Class']

for (cl, y_i) in zip(class_names, y_test):
    y_i[y_i > 0.5] = 1
    y_i[y_i < 0.5] = 0
    actives[cl] = y_i.reshape(y_i.size).astype(np.int)

for j in range(18):
    start_ind = j * 100
    end_ind = (j+1) * 100
    activations_ag = []
    activations_bin = []
    for i in range(start_ind, end_ind):

        print('Generating activation for chemical ', str(i))
        smiles = dataset.Canonical.iloc[i]
        _, hm, _ = act.activation_map(smiles, class_index=0)
        activations_ag.append(hm)
        _, hm, _ = act.activation_map(smiles, class_index=1)
        activations_bin.append(hm)
        # # _, ax = plt.subplots(figsize=(19.2, 10.1))
        # im = act.draw_smile(smiles, highlightmap=hm, cmap='coolwarm')
        # ax.imshow(im)

    dataset_j = dataset[start_ind: end_ind]
    dataset_j[class_names[0]] = activations_ag
    dataset_j[class_names[1]] = activations_bin

    dataset_j.to_csv('results/activations_' + str(start_ind)
                     + '_' + str(end_ind - 1) + '.csv')