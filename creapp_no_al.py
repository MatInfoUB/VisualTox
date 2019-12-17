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
    Classifier, Activation
figdir = os.path.join(os.getcwd(), 'creapp_figs')

X, y, class_name, new_data = load_training_data()

input_shape = X.shape[1:]

model = Classifier(epochs=1000, input_shape=input_shape, class_name=class_name)
training = model.fit(X, y, output=True)

new_data = create_new_predicted_data(new_data, X, model, class_name)
name = '2-Amino-5-azotoluene'

ind = np.where(new_data.Name == name)[0][0]
smiles = new_data.Canonical.iloc[ind]

mol = Chem.MolFromSmiles(smiles)

act = Activation(model)

fact, heatmap, heatmap_org = act.activation_map(smiles, label_index=1)
_, ax = plt.subplots(figsize=(19.2, 10.1))
im = act.draw_smile(smiles, highlightmap=heatmap, cmap='coolwarm')
ax.imshow(im)
ax.set_axis_off()

fig, ax = plt.subplots(figsize=(19.2, 10.1))
for j, (h, s) in enumerate(zip(heatmap, smiles)):
    ax.text(j, 0, s, color=cm.coolwarm(h), fontsize=40*h+16)
ax.set_xlim((0, len(smiles)))
ax.set_axis_off()
