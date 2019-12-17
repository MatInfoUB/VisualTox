from keras import backend as K
import numpy as np
from utils import Smile
from keras.preprocessing import sequence
import cv2


class Activation:

    def __init__(self, model):

        # Instatiate with a classifier
        self.model = model.model

        # Find the last conv layer
        conv_layers = [layer.name for layer in self.model.layers if 'conv' in layer.name]
        self.last_conv_layer = self.model.get_layer(conv_layers[-1])

    def compute_gradient(self, class_index=0, label_index=1):

        output = self.model.output[class_index][:, label_index]
        grads = K.gradients(output, self.last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1))
        return K.function([self.model.input], [pooled_grads, self.last_conv_layer.output[0]])

    def generate_heatmap(self, img_tensor, class_index=0, label_index=1, tol=1e-3):

        iterate = self.compute_gradient(class_index=class_index, label_index=label_index)
        pooled_grads_value, conv_layer_output_value = iterate([img_tensor])

               if np.sum(pooled_grads_value) == 0.0:
            p = len(pooled_grads_value)
            pooled_grads_value = np.ones(p)

        pooled_grads_value /= np.sum(pooled_grads_value)

        heatmap = np.average(conv_layer_output_value, weights=pooled_grads_value, axis=-1)
        # heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        if heatmap[0] > tol:
            tol = heatmap[0]
        heatmap[heatmap <= tol] = 0

        start_ind = np.where(heatmap)[0][0]

        return heatmap[start_ind:], heatmap

    def activation_map(self, smiles, class_index=0, label_index=1, tol=1e-3):

        maxlen = 130
        smi = Smile()
        smiles_x = smi.smile_to_sequence(smiles)
        smiles_x = sequence.pad_sequences([smiles_x], maxlen=maxlen)

        img_tensor = smiles_x[0].reshape(1, maxlen, 1)
        heatmap, heatmap_org = self.generate_heatmap(img_tensor, class_index=class_index, label_index=label_index, tol=tol)

        hm = cv2.resize(heatmap, (1, len(smiles)))
        hm = hm.reshape(hm.size)

        fact = float(len(heatmap)) / float(len(smiles))

        return fact, hm, heatmap_org

    def draw_smile(self, smile, highlightmap=None, cmap='coolwarm'):

        smi = Smile()
        return smi.draw_smile(smile, highlightmap=highlightmap, cmap=cmap)

