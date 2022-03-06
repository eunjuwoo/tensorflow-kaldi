from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
    
class DenseLayer1(Model):
    def __init__(self, activation='relu', num_class=4):
        super(DenseLayer1, self).__init__()
        self.dense1 = Dense(units=128, activation=activation)
        self.dense2 = Dense(units=128, activation=activation)
        self.dense3 = Dense(units=num_class+2, activation='softmax')
    
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense2(x)
        x = self.dense2(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
    