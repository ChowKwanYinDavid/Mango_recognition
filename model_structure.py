# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import keras

KERNAL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
PADDING = 'same'
ACTIVATION = 'relu'

def create_model(size, RGB):
    """
    建立CNN模型結構
    
    Args:
        size: 圖片尺寸 (e.g., 120)
        RGB: RGB通道數 (e.g., 3)
    
    Returns:
        model: 建立好的Keras模型
    """
    model = keras.Sequential()
    
    model.add(keras.layers.Conv2D(filters=8, kernel_size=KERNAL_SIZE, padding=PADDING, input_shape=(size, size, RGB), activation=ACTIVATION))
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
