
# імпортувати пакети
import keras
import numpy as np
import matplotlib.pyplot as plt

# завантажити дані з файлів
train_data_path = keras.utils.get_file('mnist_train.csv', 'file:///C:/II labs 3/Lab03/mnist_train.csv')
test_data_path = keras.utils.get_file('mnist_test.csv', 'file:///C:/II labs 3/Lab03/mnist_test.csv')

# розділити дані на вхідні та вихідні мітки
with open(train_data_path, 'r') as f:
    train_data = np.array([list(map(float, line.split(','))) for line in f.readlines()[1:] if line])
with open(test_data_path, 'r') as f:
    test_data = np.array([list(map(float, line.split(','))) for line in f.readlines()[1:] if line])

x_train, y_train = train_data[:, 1:], train_data[:, 0]
x_test, y_test = test_data[:, 1:], test_data[:, 0]

# нормалізувати дані
x_train = x_train / 255.0
x_test = x_test / 255.0

# перетворити мітки у формат one-hot
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# створити модель нейронної мережі
model = keras.models.Sequential()
model.add(keras.layers.Dense(64, activation='relu', input_shape=(784,)))
model.add(keras.layers.Dense(10, activation='softmax'))

# скомпілювати модель
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# навчити модель на тренувальних даних
model.fit(x_train, y_train, epochs=10, batch_size=32)

# оцінити модель на тестових даних
model.evaluate(x_test, y_test)

# генерувати і відображати зображення
for i in range(10):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {np.argmax(model.predict(x_test[i].reshape(1, -1)))}, Actual: {np.argmax(y_test[i])}')
    plt.show()

# зберегти модель
model.save('my_model.keras')