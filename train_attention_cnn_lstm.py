import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, Add, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

# Custom F1 Metric for binary classification
def f1(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + K.epsilon())
    precision_pos = precision(y_true, y_pred)
    recall_pos = recall(y_true, y_pred)
    precision_neg = precision((K.ones_like(y_true)-y_true), (K.ones_like(y_pred)-K.clip(y_pred, 0, 1)))
    recall_neg = recall((K.ones_like(y_true)-y_true), (K.ones_like(y_pred)-K.clip(y_pred, 0, 1)))
    f_posit = 2 * ((precision_pos * recall_pos) / (precision_pos + recall_pos + K.epsilon()))
    f_neg = 2 * ((precision_neg * recall_neg) / (precision_neg + recall_neg + K.epsilon()))
    return (f_posit + f_neg) / 2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, 'project')
CSV_PATH = os.path.join(PROJECT_DIR, 'Processed_NASDAQ.csv')
MODEL_PATH = os.path.join(PROJECT_DIR, 'nasdaq_attention_cnn_lstm.keras')
SCALER_PATH = os.path.join(PROJECT_DIR, 'nasdaq_scaler.pkl')

print("1. Loading Processed NASDAQ dataset...")
df = pd.read_csv(CSV_PATH, parse_dates=['Date'], index_col='Date')
if 'Name' in df.columns:
    del df['Name']
df = df.fillna(0)

# Create Target: 1 if next day's Close > current day's Close, else 0
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df = df.iloc[:-1] # drop last row since target is unknown

feature_cols = [c for c in df.columns if c != 'Target']
X_data = df[feature_cols].values
y_data = df['Target'].values

print(f"Dataset shape: {X_data.shape}, Target distribution: {np.bincount(y_data)}")

# Scale features
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_data)
joblib.dump(scaler, SCALER_PATH)
print(f"Scaler saved to {SCALER_PATH}")

# Construct 60-day sliding window sequences
SEQ_LEN = 60
X_seq, y_seq = [], []
for i in range(len(X_scaled) - SEQ_LEN):
    X_seq.append(X_scaled[i : i + SEQ_LEN])
    y_seq.append(y_data[i + SEQ_LEN - 1])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
print(f"Sequence Data Shape: {X_seq.shape}, Labels Shape: {y_seq.shape}")

# Train / Validation Split (80% train, 20% val)
split_idx = int(len(X_seq) * 0.8)
X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")

# 2. Build Attention-CNN-LSTM Neural Network
def build_attention_cnn_lstm(seq_len=60, feature_dim=82):
    inputs = Input(shape=(seq_len, feature_dim), name="sequence_input")
    
    # 1D Convolution for local pattern extraction
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = Dropout(0.2)(x)
    
    # Recurrent LSTM for temporal context
    lstm_out = LSTM(64, return_sequences=True)(x)
    
    # Multi-Head Self-Attention for key candle focus
    attn_out = MultiHeadAttention(num_heads=4, key_dim=16)(lstm_out, lstm_out)
    attn_out = Dropout(0.2)(attn_out)
    
    # Residual Skip Connection + Layer Normalization
    x = Add()([lstm_out, attn_out])
    x = LayerNormalization()(x)
    
    # Global Pooling + Dense Classification Head
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(1, activation='sigmoid', name="up_probability")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="Attention_CNN_LSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', f1]
    )
    return model

model = build_attention_cnn_lstm(seq_len=SEQ_LEN, feature_dim=X_train.shape[2])
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
]

print("3. Training Attention-CNN-LSTM Neural Network...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# Save trained Attention Model
model.save(MODEL_PATH)
print(f"✅ Attention-CNN-LSTM model successfully trained and saved to: {MODEL_PATH}")
