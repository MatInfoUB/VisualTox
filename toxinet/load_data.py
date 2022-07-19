import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import sequence
import seaborn as sns
import matplotlib.pyplot as plt
import os
from .utils import Smile
from rdkit import Chem

from sklearn.model_selection import train_test_split


def load_training_data(corr_plot=False, figdir=None, balanced=True, augment=False, train_size=None):

    complete_data = pd.read_excel('data/ER_trainingSet.xlsx')
    # canonical_smiles = complete_data['Canonical']
    # y_agonist = complete_data['Agonist_Class']
    # y_antagonist = complete_data['Antagonist_Class']
    # y_binding = complete_data['Binding_Class']

    classes = ['Agonist_Class', 'Antagonist_Class', 'Binding_Class']

    # 0: Inactive, 1: Active
    for cl in classes:
        complete_data[cl][complete_data[cl] == 0] = 'Inactive'
        complete_data[cl][complete_data[cl] == 1] = 'Active'

    labels = [' '.join(c.split('_')) for c in classes]
    if corr_plot:
        corr = complete_data[classes].corr()
        if figdir is None:
            raise ValueError('Please sepcify the figure directory')
        plot_corr_diag(corr, labels, figdir=figdir)

    class_name = classes[0], classes[2]

    smi = Smile()
    data = complete_data[['Name', 'Canonical', classes[0], classes[2]]]
    num_classes = complete_data[class_name[0]].nunique() + complete_data[class_name[1]].nunique()

    data_2 = complete_data[['Canonical', classes[1]]]
    num_classes_2 = 2

    if balanced:
        new_data = data[balance_data(data, class_name, class_label=1)]
    else:
        new_data = data

    new_data_train = None
    new_data_test = None

    if train_size is not None:
        idx_train, idx_test = train_test_split(range(len(new_data)), train_size=train_size)
        new_data_train = new_data.iloc[idx_train]
        new_data_test = new_data.iloc[idx_test]

    if augment:
        if train_size is None:
            new_data = data_augmenter(new_data, num_generator=10)
        else:
            new_data_train = data_augmenter(new_data_train, num_generator=10)
            new_data_test = data_augmenter(new_data_test, num_generator=10)

    if train_size is None:
        X = smi.smiles_to_sequences(new_data.Canonical, embed=False)
        maxlen = 130
        X = sequence.pad_sequences(X, maxlen=maxlen)
        X = X / smi.max_num
        X = X.reshape(len(X), maxlen, 1)

        # y_1 = pd.get_dummies(new_data[class_name[0]])
        # y_2 = pd.get_dummies(new_data[class_name[1]])

        y = [new_data[cl] for cl in class_name]
        return X, y, class_name, new_data
    else:
        X_train = smi.smiles_to_sequences(new_data_train.Canonical, embed=False)
        maxlen = 130
        X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
        X_train = X_train / smi.max_num
        X_train = X_train.reshape(len(X_train), maxlen, 1)
        y_train = [new_data_train[cl] for cl in class_name]

        X_test = smi.smiles_to_sequences(new_data_test.Canonical, embed=False)
        maxlen = 130
        X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
        X_test = X_test / smi.max_num
        X_test = X_test.reshape(len(X_test), maxlen, 1)
        y_test = [new_data_test[cl] for cl in class_name]

        return X_train, X_test, y_train, y_test, class_name, new_data_train, new_data_test


def extract_x_y(data, class_names):

    smi = Smile()
    X = smi.smiles_to_sequences(data.Canonical, embed=False)

    maxlen = 130

    X = sequence.pad_sequences(X, maxlen=maxlen)
    X = X.astype(np.float32) / (np.float32(smi.max_num))
    X = X.reshape(len(X), maxlen, 1)

    y_1 = data[class_names[0]]
    y_2 = data[class_names[1]]

    y = [y_1, y_2]

    return X, y


def balance_data(data, class_name, class_label):

    np.random.seed(0)
    n_samp = data[class_name[class_label]].value_counts().min()
    inds = np.zeros(len(data), dtype='bool')
    for label in data[class_name[class_label]].value_counts().index:
        ind = np.where(data[class_name[class_label]] == label)[0]
        if len(ind) == n_samp:
            inds[ind] = True
        else:
            ii = np.random.choice(ind, n_samp, replace=False)
            inds[ii] = True

    return inds


def plot_corr_diag(corr, labels, figdir):

    g = sns.heatmap(corr, xticklabels=labels, yticklabels=labels,
                cmap='coolwarm', annot=True, cbar=False, annot_kws={"size": 20})
    # g.set_xticklabels(g.get_xticklabels(), rotation=15)
    g.set_yticklabels(g.get_yticklabels(), rotation=0)

    plt.savefig(os.path.join(figdir, 'correlation.png'), bbox_inches='tight', dpi=300)


def create_new_predicted_data(old_data, X, model, class_name):

    y_pred = model.predict(X)
    y_pred = [class_y.argmax(axis=-1) for class_y in y_pred]

    old_data[class_name[0] + '_predicted'] = y_pred[0]
    old_data[class_name[1] + '_predicted'] = y_pred[1]

    old_data['Length'] = old_data.Canonical.apply(lambda x: len(x))
    new_data = old_data.sort_values(by='Length', ascending=False)

    return new_data


def imbalance_plot(data, figdir):

    classes = ['Agonist_Class', 'Antagonist_Class', 'Binding_Class']

    fig, axes = plt.subplots(nrows=1, ncols=3)
    explode = (0, 0.1)

    for (ax, cl) in zip(axes.flat, classes):
        name = ' '.join(cl.split('_'))
        S = pd.Series(data[cl], name=name)
        S.value_counts().plot.pie(explode=explode, ax=ax, startangle=90, autopct='%1.1f%%',
                                  textprops=dict(color="w"))
        ax.axis('equal')

    plt.savefig(os.path.join(figdir, 'imbalace.png'), bbox_inches='tight', dpi=300)


def load_evaluation_data(class_name=None, balanced=False, augment=False, train_size=None):

    new_data = pd.read_csv('data/0422_evaluation_set_agonist&binding.csv').drop('Unnamed: 0', axis=1)
    # ind = [',' not in name for name in complete_data.Name]
    #
    # complete_data = complete_data[ind]
    # complete_data = complete_data[['Mode', 'COMPOUND_NAME', 'STANDERDIZED_CANO_SMI']]


    for cl in class_name:
        new_data[cl][new_data[cl] == 0] = 'Inactive'
        new_data[cl][new_data[cl] == 1] = 'Active'


    if balanced:
        new_data = new_data[balance_data(new_data, class_name, class_label=0)]
    else:
        new_data = new_data

    new_data_train = None
    new_data_test = None

    if train_size is not None:
        idx_train, idx_test = train_test_split(range(len(new_data)), train_size=train_size)
        new_data_train = new_data.iloc[idx_train]
        new_data_test = new_data.iloc[idx_test]

    if augment:
        if train_size is None:
            new_data = data_augmenter(new_data, num_generator=10)
        else:
            new_data_train = data_augmenter(new_data_train, num_generator=10)
            new_data_test = data_augmenter(new_data_test, num_generator=10)

    smi = Smile()
    if train_size is None:
        X = smi.smiles_to_sequences(new_data.Canonical, embed=False)
        maxlen = 130
        X = sequence.pad_sequences(X, maxlen=maxlen)
        X = X / smi.max_num
        X = X.reshape(len(X), maxlen, 1)

        # y_1 = pd.get_dummies(new_data[class_name[0]])
        # y_2 = pd.get_dummies(new_data[class_name[1]])

        y = [new_data[cl] for cl in class_name]
        return X, y, class_name, new_data
    else:
        X_train = smi.smiles_to_sequences(new_data_train.Canonical, embed=False)
        maxlen = 130
        X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
        X_train = X_train / smi.max_num
        X_train = X_train.reshape(len(X_train), maxlen, 1)
        y_train = [new_data_train[cl] for cl in class_name]

        X_test = smi.smiles_to_sequences(new_data_test.Canonical, embed=False)
        maxlen = 130
        X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
        X_test = X_test / smi.max_num
        X_test = X_test.reshape(len(X_test), maxlen, 1)
        y_test = [new_data_test[cl] for cl in class_name]

        return X_train, X_test, y_train, y_test, class_name, new_data_train, new_data_test


def load_prediction_data():

    complete_data = pd.read_excel('data/ER_predictionSet.xlsx')
    complete_data = complete_data[['CHEMICAL NAME', 'Canonical_SMI']]

    inds = [isinstance(d, str) for d in complete_data['CHEMICAL NAME']]
    complete_data = complete_data[inds]

    complete_data['CHEMICAL NAME'] = complete_data['CHEMICAL NAME'].apply(lambda x:x.split('|')[0])

    new_data = pd.DataFrame({'Name': complete_data['CHEMICAL NAME'], 'Canonical': complete_data['Canonical_SMI']})
    new_data['Smile_length'] = new_data['Canonical'].apply(lambda x:len(x))

    new_data = new_data[new_data['Smile_length'] < 130]

    new_data = new_data.sort_values(by='Smile_length', ascending=False)

    smi = Smile()
    X = smi.smiles_to_sequences(new_data.Canonical, embed=False)

    maxlen = 130

    X = sequence.pad_sequences(X, maxlen=maxlen)
    X = X.astype(np.float32) / (np.float32(smi.max_num))
    X = X.reshape(len(X), maxlen, 1)

    return X, new_data


def smiles_augmenter(smiles, num_generator=10, shuffle_limit=1000):

    mol = Chem.MolFromSmiles(smiles)
    num_atoms = mol.GetNumAtoms()

    smiles_set = []
    if num_atoms < 4:
        from itertools import permutations
        perms = list(permutations(range(num_atoms)))
        for p in perms:
            smiles_set.append(
                Chem.MolToSmiles(Chem.RenumberAtoms(mol, p),
                                 canonical=False, isomericSmiles=True))

        return smiles_set

    count = 0
    while len(smiles_set) < num_generator:
        p = np.random.permutation(range(num_atoms))
        new_smiles = Chem.MolToSmiles(
            Chem.RenumberAtoms(mol, p.tolist()),
            canonical=False, isomericSmiles=True)
        if new_smiles not in smiles_set:
            smiles_set.append(new_smiles)

        count += 1
        if count == shuffle_limit:
            break

    return smiles_set


def data_augmenter(data, num_generator=10):

    smiles_new = []
    names_new = []
    y1_new = []
    y2_new = []

    for i in range(len(data)):
        datum = data.iloc[i]
        name = datum['Name']
        smiles = datum['Canonical']
        ag_cl = datum['Agonist_Class']
        bi_cl = datum['Binding_Class']

        smiles_set = smiles_augmenter(smiles, num_generator=num_generator)
        y1_new.append(pd.Series([ag_cl] * len(smiles_set)))
        y2_new.append(pd.Series([bi_cl] * len(smiles_set)))
        smiles_new.append(pd.Series(smiles_set))
        names_new.append(pd.Series([name] * len(smiles_set)))

    return pd.DataFrame({'Name': pd.concat(names_new),
                         'Canonical':pd.concat(smiles_new),
                         'Agonist_Class': pd.concat(y1_new),
                         'Binding_Class': pd.concat(y2_new)})


def combined_data():

    _, _, class_name, train_data = load_training_data(balanced=True, augment=True)
    _, _, val_data = load_evaluation_data(class_name, balanced=True, augment=True)

    return extract_x_y(train_data.append(val_data), class_name)