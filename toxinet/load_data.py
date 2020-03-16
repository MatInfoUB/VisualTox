import pandas as pd
import numpy as np
from keras.preprocessing import sequence
import seaborn as sns
import matplotlib.pyplot as plt
import os
from .utils import Smile


def load_training_data(corr_plot=False, figdir=None, balanced=True):

    complete_data = pd.read_excel('data/ER_trainingSet.xlsx')
    canonical_smiles = complete_data['Canonical']
    y_agonist = complete_data['Agonist_Class']
    y_antagonist = complete_data['Antagonist_Class']
    y_binding = complete_data['Binding_Class']

    classes = ['Agonist_Class', 'Antagonist_Class', 'Binding_Class']

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
        n_samp = data[class_name[1]].value_counts().min()
        inds = []
        for label in data[class_name[1]].value_counts().index:
            ind = np.where(data[class_name[1]] == label)[0]
            if len(ind) == n_samp:
                inds.append(data.loc[ind])
            else:
                inds.append(data.loc[np.random.choice(ind, n_samp)])

        new_data = pd.concat(inds)
    else:
        new_data = data
    X = smi.smiles_to_sequences(new_data.Canonical, embed=False)

    maxlen = 130
    X = sequence.pad_sequences(X, maxlen=maxlen)
    X = X.astype(np.float32) / (np.float32(smi.max_num))
    X = X.reshape(len(X), maxlen, 1)

    y_1 = pd.get_dummies(new_data[class_name[0]])
    y_2 = pd.get_dummies(new_data[class_name[1]])

    y = [y_1, y_2]

    return X, y, class_name, new_data


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
        S.value_counts().plot.pie(explode=explode, ax=ax, startangle=90,
           autopct='%1.1f%%')
        ax.axis('equal')

    plt.savefig(os.path.join(figdir, 'imbalace.png'), bbox_inches='tight', dpi=300)


def load_evaluation_data(class_name=None):

    complete_data = pd.read_excel('data/ER_evaluationSet.xlsx')
    ind = [',' not in name for name in complete_data.COMPOUND_NAME]

    complete_data = complete_data[ind]
    complete_data = complete_data[['Mode', 'COMPOUND_NAME', 'STANDERDIZED_CANO_SMI']]

    complete_data['COMPOUND_NAME'] = complete_data['COMPOUND_NAME'].apply(lambda x: x[:-3])
    complete_data['COMPOUND_NAME'] = complete_data['COMPOUND_NAME'].str.title()

    table = np.zeros((complete_data['STANDERDIZED_CANO_SMI'].nunique(), 3), dtype=np.int8)
    # Names = complete_data['COMPOUND_NAME'].unique().tolist()
    Smiles = pd.Series(complete_data['STANDERDIZED_CANO_SMI'].unique())

    from sklearn.preprocessing import LabelEncoder
    classes = LabelEncoder().fit_transform(complete_data['Mode'])

    Names = []
    Smiles_completed = []
    for i, cl in enumerate(classes):
        smi = complete_data['STANDERDIZED_CANO_SMI'].iloc[i]
        ind = Smiles.to_list().index(smi)
        table[ind, cl] = 1

        name = complete_data['COMPOUND_NAME'].iloc[i]
        if smi not in Smiles_completed:
            Smiles_completed.append(smi)
            Names.append(name)

    Names = pd.Series(Names)
    new_data = pd.DataFrame({'Name': Names, 'Canonical': Smiles, 'Agonist_Class': table[:, 0],
                             'Binding_Class': table[:, 2]})


    smi = Smile()
    X = smi.smiles_to_sequences(new_data.Canonical, embed=False)

    maxlen = 130

    X = sequence.pad_sequences(X, maxlen=maxlen)
    X = X.astype(np.float32) / (np.float32(smi.max_num))
    X = X.reshape(len(X), maxlen, 1)

    y_1 = pd.get_dummies(new_data[class_name[0]])
    y_2 = pd.get_dummies(new_data[class_name[1]])

    y = [y_1, y_2]

    return X, y, new_data

def load_prediction_data(class_name=None):

    complete_data = pd.read_excel('data/ER_predictionSet.xlsx')
    complete_data = complete_data[['CHEMICAL NAME', 'Canonical_SMI']]

    inds = [isinstance(d, str) for d in complete_data['CHEMICAL NAME']]
    complete_data = complete_data[inds]

    complete_data['CHEMICAL NAME'] = complete_data['CHEMICAL NAME'].apply(lambda x:x.split('|')[0])

    new_data = pd.DataFrame({'Name': complete_data['CHEMICAL NAME'], 'Canonical': complete_data['Canonical_SMI']})

    smi = Smile()
    X = smi.smiles_to_sequences(new_data.Canonical, embed=False)

    maxlen = 130

    X = sequence.pad_sequences(X, maxlen=maxlen)
    X = X.astype(np.float32) / (np.float32(smi.max_num))
    X = X.reshape(len(X), maxlen, 1)

    return X, new_data