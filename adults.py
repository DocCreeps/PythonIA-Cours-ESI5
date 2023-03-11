import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Charger le jeu de données
df = pd.read_csv('adults3.csv', header=0)
df = df.fillna(df.mean())

# Afficher le dataframe pour vérifier si la colonne 'fnlwgt' a été supprimée
print(df.columns)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encoder la variable de sortie
le = LabelEncoder()
y = le.fit_transform(y)

# Transformer les variables d'entrée
ct = make_column_transformer(
    (OneHotEncoder(), [ 'workclass','education', 'marital-status', 'occupation',
                       'relationship', 'race', 'sex', 'native-country']),
    remainder='passthrough')
X = ct.fit_transform(X)

# Diviser le jeu de données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Centrer et réduire les données
scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Créer le réseau de neurones
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compiler le modèle
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)

score = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: ", score[1])

# Prédire la classe pour un individu aléatoire
new_data = pd.DataFrame({
    'age': [37],
    'workclass': ['State-gov'],
    'fnlwgt': [10000],
    'education': ['Some-college'],
    'education-num': [10],
    'marital-status': ['Married-civ-spouse'],
    'occupation': ['Exec-managerial'],
    'relationship': ['Husband'],
    'race': ['White'],
    'sex': ['Male'],
    'capitalgain': [0],
    'capitalloss': [0],
    'hoursperweek': [80],
    'native-country': ['United-States'],

    
})

# Encoder les variables catégorielles
new_data = ct.transform(new_data)

# Standardiser les données
new_data = scaler.transform(new_data)
# Faire une prédiction
prediction = model.predict(new_data)

# Afficher la prédiction
print(prediction)

