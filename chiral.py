import pandas as pd
import numpy as np
from keras.preprocessing import sequence
import os

data = pd.read_csv('data/chiralPairs_ChEMBL23_final_filtered.csv')

input_col = 'rdkitNonIsoCanonSmi'
output_col = 'Chiral cliff'

X = data[input_col]
inds = np.where(np.asarray(["'" in s for s in X]))[0]
X = X.drop(X.index[inds])

from toxinet import Smile
smi = Smile()
X = smi.smiles_to_sequences(X, embed=False)
maxlen = 200
X = sequence.pad_sequences(X, maxlen=maxlen)
X = X.reshape(len(X), maxlen, 1)

y = data[output_col]
y = y.drop(y.index[inds])

from sklearn.preprocessing import LabelEncoder

y_true = LabelEncoder().fit_transform(y)
y = pd.get_dummies(y)

input_shape = X.shape[1:]

from toxinet import Classifier, diagnostics
model = Classifier(epochs=1000, input_shape=input_shape, num_output=1)
keras_model = model.model

model.training = keras_model.fit(X, y, verbose=1, batch_size=100, epochs=model.epochs)

model_json = keras_model.to_json()
import json
json.dump(model_json, open('chiral_results/model.json', 'w'))
model_weights = keras_model.save_weights(filepath='chiral_results/model_weights.h5')

y_pred = keras_model.predict(X)
y_pred = [class_y.argmax(axis=-1) for class_y in y_pred]
X_new = data[input_col]
X_new = X_new.drop(X_new.index[inds])
new_data = pd.DataFrame({'Name': X_new, 'Observed': y_true, 'Predicted': y_pred})
new_data['SUM'] = new_data.Observed + new_data.Predicted

chirals = new_data[new_data['SUM'] == 0]
chirals['length'] = chirals['Name'].apply(lambda x: len(x))
chirals = chirals.sort_values(by='length', ascending=False)

import matplotlib.pyplot as plt

_, ax = plt.subplots()
ax.plot(model.training.history['accuracy'])
ax.set_xlabel('Iteration')
ax.set_ylabel('Training Accuracy')
plt.savefig('chiral_results/training.png', bbox_inches='tight', dpi=300)

import keras.backend as K

# Diagnostics
conv_layers = [layer.name for layer in keras_model.layers if 'conv' in layer.name]
last_conv_layer = keras_model.get_layer(conv_layers[-1])
iterate = K.function([keras_model.input], [last_conv_layer.output[0]])

from rdkit import Chem
from scipy.interpolate import interp1d
import matplotlib.cm as cm

fig_folder = os.path.join('chiral_results', 'chiral_activations')

tol = 1e-3
for i in range(30):
    smiles = chirals.Name.iloc[i]
    mol = Chem.MolFromSmiles(smiles)

    smi = Smile()
    smiles_x = smi.smile_to_sequence(smiles)
    smiles_x = sequence.pad_sequences([smiles_x], maxlen=maxlen)

    img_tensor = smiles_x[0].reshape(1, maxlen, 1)
    conv_layer_output_value = iterate([img_tensor])[0]


    heatmap = np.average(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    if heatmap[0] > tol:
        tol = heatmap[0]
    heatmap[heatmap <= tol] = 0
    start_ind = np.where(heatmap)[0][0]

    heatmap = heatmap[start_ind:]

    x_org = np.linspace(0, 1, len(heatmap))
    x_new = np.linspace(0, 1, len(smiles))
    f1 = interp1d(x_org, heatmap, kind='nearest')
    heatmap_smiles = f1(x_new)

    _, ax = plt.subplots(figsize=(19.2, 10.1))
    im = smi.draw_smile(smiles, highlightmap=heatmap_smiles, cmap='coolwarm')
    ax.imshow(im)
    ax.set_axis_off()
    plt.savefig(os.path.join(fig_folder, 'chemical_' + str(i)+ '.png'), bbox_inches='tight', dpi=300)
    plt.close()

    _, ax = plt.subplots(figsize=(19.2, 10.1))
    for j, (h, s) in enumerate(zip(heatmap_smiles, smiles)):
        ax.text(j, 0, s, color=cm.coolwarm(h), fontsize=40 * h + 16)
    ax.set_xlim((0, len(smiles)))
    ax.set_axis_off()
    plt.savefig(os.path.join(fig_folder, 'chemical_' + str(i) + '_activations.png'), bbox_inches='tight', dpi=300)
    plt.close()


