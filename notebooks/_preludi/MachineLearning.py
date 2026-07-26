import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

rng = np.random.default_rng(0)

# Il dataset che il capitolo dà per esistente: nelle pagine il punto è il
# modello e la metrica, non da dove vengono i numeri.
X, y = make_classification(n_samples=400, n_features=8, n_informative=5,
                           n_redundant=1, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

# Predizioni pronte: le pagine sulle metriche le usano senza calcolarle.
_albero = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)
y_pred = _albero.predict(X_test)
y_prob = _albero.predict_proba(X_test)[:, 1]

# I nomi concreti con cui il libro racconta gli esempi (prezzi, spam, un input
# nuovo da predire) qui esistono, con numeri finti.
y_prezzo = 150_000 + 12_000 * X_train[:, 0] + rng.normal(0, 5_000, len(X_train))
y_spam = y_train
X_nuovo = X_test[:3]

# Dati "di produzione" per la pagina sul distribution shift: gli stessi input
# con la prima feature spostata, che è esattamente il guasto che quel capitolo
# insegna a scoprire.
X_prod = X_test.copy()
X_prod[:, 0] += 1.5
