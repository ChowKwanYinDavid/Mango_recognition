import keras

# TODO: set it at main.py / constants.py
KERNAL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
PADDING = 'same'
ACTIVATION = 'relu'

class CNNModel:
    def __init__(self, size, RGB):
        self.size = size
        self.RGB = RGB
        self.model = self.create_model()

    def create_model(self) -> keras.Sequential:
        """
        建立CNN模型結構
        
        Returns:
            model: 建立好的Keras模型
        """
        model = keras.Sequential()
        
        model.add(keras.layers.Conv2D(filters=8, kernel_size=KERNAL_SIZE, padding=PADDING, input_shape=(self.size, self.size, self.RGB), activation=ACTIVATION))
        model.add(keras.layers.MaxPooling2D(pool_size=POOL_SIZE))
        model.add(keras.layers.Conv2D(filters=16, kernel_size=KERNAL_SIZE, padding=PADDING, activation=ACTIVATION))
        model.add(keras.layers.MaxPooling2D(pool_size=POOL_SIZE))
        model.add(keras.layers.Conv2D(filters=32, kernel_size=KERNAL_SIZE, padding=PADDING, activation=ACTIVATION))
        model.add(keras.layers.MaxPooling2D(pool_size=POOL_SIZE))
        model.add(keras.layers.Conv2D(filters=64, kernel_size=KERNAL_SIZE, padding=PADDING, activation=ACTIVATION))
        model.add(keras.layers.MaxPooling2D(pool_size=POOL_SIZE))
        model.add(keras.layers.Conv2D(filters=128, kernel_size=KERNAL_SIZE, padding=PADDING, activation=ACTIVATION))
        model.add(keras.layers.MaxPooling2D(pool_size=POOL_SIZE))
        model.add(keras.layers.Conv2D(filters=128, kernel_size=KERNAL_SIZE, padding=PADDING, activation=ACTIVATION))
        model.add(keras.layers.MaxPooling2D(pool_size=POOL_SIZE))
        
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(5, activation='softmax'))
        
        return model