# %%
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# %%
print(tf.__version__)

# %%
df = pd.read_csv('sample.csv')

# %%
X_data = df[['sex', 'AGE', 'BMI', 'MVPA', 'rest_HR']]
y_data = df['VO2max']

X_data['sex'] = X_data['sex'] * 1
X_data['MVPA'] = X_data['MVPA'] * 1

X_data = X_data.values
y_data = y_data.values

# %%
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
                                                    test_size=0.2,
                                                    random_state=1004)

# %%
feature_num = X_data.shape[-1]
layer_num = 2

inputs = tf.keras.layers.Input(shape=(feature_num,))
x = inputs

for i in range(2):
    x = tf.keras.layers.Dense(100, use_bias=False)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.BatchNormalization()(x)

outputs = tf.keras.layers.Dense(1)(x)

model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer = 'adam',
    loss = 'mean_squared_error'
)

# %%
model.summary()

# %%
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=128
)
# %%
