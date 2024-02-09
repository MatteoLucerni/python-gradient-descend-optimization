import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.datasets import make_classification
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


X, Y = make_classification(n_samples=1250, n_features=4, n_informative=2, random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

sgd = SGDClassifier(loss='log_loss')
sgd.fit(X_train, Y_train)

print(f'Loss: {log_loss(Y_test, sgd.predict_proba(X_test))}')

def minibatch(train_set, test_set, n_batches, epochs):
    X_train, Y_train = train_set
    X_test, Y_test = test_set

    batch_size = X_train.shape[0]/n_batches
    sgd = SGDClassifier(loss='log_loss')
    sgd_loss = []

    best_loss = 1

    for epoch in range(epochs):
        # mischio il dataset
        X_shuffle, Y_shuffle = shuffle(X_train, Y_train)
        
        for batch in range(n_batches):

            classes = np.unique(Y_train)

            batch_start = int(batch * batch_size)
            batch_end = int((batch + 1) * batch_size)

            X_batch = X_shuffle[batch_start:batch_end,:]
            Y_batch = Y_shuffle[batch_start:batch_end]

            sgd.partial_fit(X_batch, Y_batch, classes=classes)
            loss = log_loss(Y_test, sgd.predict_proba(X_test), labels=classes)
            sgd_loss.append(loss)
        
        # print(f'Loss all epoca {epoch}: {loss}')

        if loss < best_loss:
            best_loss = loss
    
    return (sgd, sgd_loss, best_loss)

# full batch gradient descend
full_gd, full_gd_loss, full_best_loss = minibatch((X_train, Y_train), (X_test, Y_test), n_batches=1, epochs=200)

# stochastic gradient descend
stoch_gd, stoch_gd_loss, stoch_best_loss = minibatch((X_train, Y_train), (X_test, Y_test), n_batches=X_train.shape[0], epochs=5)

# mini batch gradient descend
mini_batch_gd, mini_batch_gd_loss, mini_batch_best_loss = minibatch((X_train, Y_train), (X_test, Y_test), n_batches=10, epochs=50)

print(f'Full Batch Loss: {full_best_loss} / Stochastic Loss: {stoch_best_loss} / Mini Batch Loss: {mini_batch_best_loss}')

plt.rcParams['figure.figresize']=(14,10)

plt.plot(full_gd_loss, label='Full Batch')


plt.plot(stoch_gd_loss, label='Stochastic')


plt.plot(mini_batch_gd_loss, label='Mini Batch')

plt.xlim(xmin=0, xmax=200)
plt.legend()
plt.show()