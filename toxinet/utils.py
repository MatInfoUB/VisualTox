from rdkit import Chem, DataStructs
#from gensim.models import Word2Vec
import numpy as np
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import DrawingOptions

from matplotlib.colors import ColorConverter as cc
import matplotlib.cm as cm


class Smile:

    def __init__(self):

        self.SMILES_CHARS = [' ',
                        '#', '%', '(', ')', '+', '-', '.', '/', ':',
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                        '=', '@',
                        'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
                        'R', 'S', 'T', 'V', 'X', 'Z',
                        '[', '\\', ']',
                        'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
                        't', 'u']
        self.max_num = len(self.SMILES_CHARS)
        self.smi2index = dict((c, i) for i, c in enumerate(self.SMILES_CHARS))
        self.index2smi = dict((i, c) for i, c in enumerate(self.SMILES_CHARS))

    # def train_smiles(self, smile_file='data/smiles'):
    #
    #     smiles = open(smile_file, 'rb').read().split('\n')
    #     smiles.pop()
    #     can_smi = [Chem.MolToSmiles(Chem.MolFromSmiles(smi),
    #                                 isomericSmiles=True, canonical=True) for smi in smiles]
    #     model = Word2Vec([can_smi], size=100, window=5, min_count=1)
    #     self.model = model

    def smile_to_sequence(self, smile, embed=False, maxlen=120):

        inds = np.asarray([self.smi2index[s] for s in smile])
        if not embed:
            return inds
        ex = maxlen - len(inds)
        return np.r_[np.zeros(ex/2), inds, np.zeros(ex - ex/2)]

    def smiles_to_sequences(self, smiles, embed=False, maxlen=120):

        return np.asarray([self.smile_to_sequence(smile, embed=embed, maxlen=maxlen) for smile in smiles])

    def smile_to_vector(self, smiles):

        can_smi = [Chem.MolToSmiles(Chem.MolFromSmiles(smi),
                                    isomericSmiles=True, canonical=True) for smi in smiles]

        return [self.model.wv(smile).reshape(10, 10) for smile in can_smi]

    def smile_encoder(self, smile, maxlen=120):

        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smile))
        X = np.zeros( ( maxlen, len( self.SMILES_CHARS ) ) )
        for i, c in enumerate( smiles ):
            X[i, self.smi2index[c] ] = 1
        return X

    def smiles_to_hot(self, smiles, maxlen=120):

        can_smi = [Chem.MolToSmiles(Chem.MolFromSmiles(smi),
                                    isomericSmiles=True, canonical=True) for smi in smiles]

        return [self.smile_encoder(smi, maxlen=maxlen) for smi in can_smi]

    def compare_smile(self, smile1, smile2):
        mol1 = Chem.MolFromSmiles(smile1)
        mol2 = Chem.MolFromSmiles(smile2)

        # fp1 = Chem.RDKFingerprint(mol1)
        # fp2 = Chem.RDKFingerprint(mol2)
        mfp1 = rdMolDescriptors.GetMorganFingerprint(mol1, 2)
        mfp2 = rdMolDescriptors.GetMorganFingerprint(mol2, 2)

        return DataStructs.TanimotoSimilarity(mfp1, mfp2)

    def draw_smile(self, smile, highlightmap=None, cmap='coolwarm'):

        mol = Chem.MolFromSmiles(smile)
        opts = DrawingOptions()
        opts.atomLabelFontSize = 110
        opts.dotsPerAngstrom = 200
        opts.bondLineWidth = 6.0

        if highlightmap is None:
            im = Chem.Draw.MolToImage(mol, size=(3840, 2160), options=opts)
        else:
            colors = {}
            for k, h in enumerate(highlightmap):
                colors[k] = cc.to_rgb(cm.get_cmap(cmap)(h))

            im = Chem.Draw.MolToImage(mol, size=(3840, 2160), options=opts, highlightMap=colors)

        return im
