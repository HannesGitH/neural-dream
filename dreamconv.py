from tf2onnx.convert import from_keras
import tensorflow as tf

from tensorflow.keras.layers import *


orgmodel = tf.keras.models.load_model('models/dreamr')

input = Input(shape=(3, 640, 480))
x = input
x = Permute((2,3,1))(x)
x = orgmodel(x)
x = Reshape((3, 672, 627))(x)
preds = x

model = tf.keras.Model(input,preds)

spec = (tf.TensorSpec((1, 3, 640, 480), tf.float32, name="input"),)
model_proto, external_tensor_storage = from_keras(model, input_signature=spec,output_path='models/dreamr.onnx')