from toxinet import ConvLSTMToxinet
from toxinet.load_data import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import load_model

X, y, class_name, new_data = load_training_data(balanced=True, augment=True)
lbe = LabelEncoder().fit(y[0])
y_1 = lbe.transform(y[0])
y_2 = lbe.transform(y[1])

X_train, X_test, y_1_train, y_1_test, y_2_train, y_2_test = \
    train_test_split(X, y_1, y_2, test_size=0.3, random_state=0)


input_shape = X.shape[1:]

model = ConvLSTMToxinet(epochs=3000, input_shape=input_shape,
                   class_name=class_name, batch_size=2000, learning_rate=0.005)
y_test = [y_1_test, y_2_test]
model.model = load_model('results/al_model_lstm.h5')

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import classification_report

fig1, ax = plt.subplots()
fig2, fax = plt.subplots()
fig_hm, axes = plt.subplots(nrows=1, ncols=2)

for i in range(2):
    y_score = model.predict(X_test)[i]
    y_i_test = y_test[i]
    fpr, tpr, th = roc_curve(y_i_test, y_score)
    print('AUC value for', class_name[i], 'is :', auc(fpr, tpr))
    ax.plot(fpr, tpr, lw=2, label=class_name[i] + ' AUC: ' + "{:.3f}".format(auc(fpr, tpr)))

    precision, recall, _ = precision_recall_curve(y_i_test, y_score)
    fax.plot(precision, recall, lw=2, label=class_name[i] + ' AP: '
                                            + "{:.3f}".format(average_precision_score(y_i_test, y_score)))

    dif = [tpr - fpr]
    y_test_class = y_score
    y_test_class[y_test_class > th[np.argmax(dif)]] = 1
    y_test_class = y_test_class.astype(np.int).reshape(y_test_class.size)
    cm = pd.crosstab(lbe.inverse_transform(y_i_test), lbe.inverse_transform(y_test_class))

    sns.heatmap(data=cm, annot=True, fmt="d", ax=axes.flat[i])
    axes.flat[i].set_ylabel('')
    axes.flat[i].set_xlabel('')
    print(classification_report(y_i_test, y_test_class, target_names=lbe.classes_))

    print('Final Testing Accuracy for', class_name[i], 'is :',
          np.trace(cm.values)/np.sum(cm.values))

ax.legend()
ax.plot([0, 1], [0, 1], '--', lw=2)
ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-0.05, 1.05])
ax.set_ylabel('True Positive Rate')
ax.set_xlabel('False Positive Rate')

fax.legend()
fax.set_ylabel('Precision')
fax.set_xlabel('Recall')

figdir = 'figs/active_learning/'

fig1.savefig(figdir + 'ROC_AUC_al', bbox_inches='tight', dpi=300)
fig2.savefig(figdir + 'PR_AUC_al', bbox_inches='tight', dpi=300)
fig_hm.savefig(figdir + 'Confusion_matrix_al', bbox_inches='tight', dpi=300)