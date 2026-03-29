import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import math
import collections

file_name = 'PS_20174392719_1491204439457_log.csv'

freq_map = collections.defaultdict(int)
for chunk in pd.read_csv(file_name, chunksize=200000, usecols=['nameOrig']):
    counts = chunk['nameOrig'].value_counts()
    for name, count in counts.items():
        freq_map[name] += count

sample_df = pd.read_csv(file_name, nrows=200000)
sample_df['hour_of_day'] = sample_df['step'] % 24
sample_df['transaction_freq'] = sample_df['nameOrig'].map(freq_map)

sample_df['errorBalanceOrg'] = sample_df['newbalanceOrig'] + sample_df['amount'] - sample_df['oldbalanceOrg']
sample_df['errorBalanceDest'] = sample_df['oldbalanceDest'] + sample_df['amount'] - sample_df['newbalanceDest']

features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 
            'hour_of_day', 'transaction_freq', 'errorBalanceOrg', 'errorBalanceDest', 'type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 
                                   'hour_of_day', 'transaction_freq', 'errorBalanceOrg', 'errorBalanceDest']),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['type'])])

preprocessor.fit(sample_df[features])
del sample_df 

def fraud_data_generator(file_path, chunk_size, preprocessor, freq_dict):
    while True:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, usecols=['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']):
            chunk = chunk[chunk['isFraud'] == 0]
            if len(chunk) == 0: continue

            chunk['hour_of_day'] = chunk['step'] % 24
            chunk['transaction_freq'] = chunk['nameOrig'].map(freq_dict)
            
            chunk['errorBalanceOrg'] = chunk['newbalanceOrig'] + chunk['amount'] - chunk['oldbalanceOrg']
            chunk['errorBalanceDest'] = chunk['oldbalanceDest'] + chunk['amount'] - chunk['newbalanceDest']
            
            X_processed = preprocessor.transform(chunk[features])
            yield (X_processed, X_processed)

input_dim = 14 

input_layer = Input(shape=(input_dim,))
encoder = Dense(32, activation="relu")(input_layer)
encoder = Dense(16, activation="relu")(encoder)
encoder = Dense(6, activation="relu")(encoder)

decoder = Dense(16, activation="relu")(encoder)
decoder = Dense(32, activation="relu")(decoder)
decoder = Dense(input_dim, activation="linear")(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

print("Started learning...")
chunk_size = 50000
total_normal_rows = 6354407
steps_per_epoch = math.ceil(total_normal_rows / chunk_size)

train_gen = fraud_data_generator(file_name, chunk_size, preprocessor, freq_map)
early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

history = autoencoder.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=15, 
    callbacks=[early_stop],
    verbose=1
)

test_df = pd.read_csv(file_name, usecols=['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud'])
test_df['hour_of_day'] = test_df['step'] % 24
test_df['transaction_freq'] = test_df['nameOrig'].map(freq_map)
test_df['errorBalanceOrg'] = test_df['newbalanceOrig'] + test_df['amount'] - test_df['oldbalanceOrg']
test_df['errorBalanceDest'] = test_df['oldbalanceDest'] + test_df['amount'] - test_df['newbalanceDest']

fraud_data = test_df[test_df['isFraud'] == 1]
normal_data = test_df[test_df['isFraud'] == 0].sample(n=500000, random_state=42)

test_set = pd.concat([normal_data, fraud_data])
X_test = preprocessor.transform(test_set[features])
y_test = test_set['isFraud'].values

predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)

threshold = np.percentile(mse[y_test == 0], 99) 
y_pred = [1 if e > threshold else 0 for e in mse]

print("\n--- FINAL RESULTS ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
