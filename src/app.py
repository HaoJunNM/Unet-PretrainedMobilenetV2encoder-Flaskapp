
import tensorflow as tf
from flask import Flask, request
import numpy as np
import keras
from keras.models import load_model

app = Flask(__name__)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

global graph
graph = tf.compat.v1.get_default_graph()
model = load_model('models1.h5',custom_objects={'dice_loss': dice_loss, 'dice_coef':dice_coef})

@app.route('/inference', methods=["POST"])
def inference():
    data = request.json
    img_arr = np.array(data["image"], dtype=np.uint8)
    
    # TODO: replace with call to your model and any other code
    #random_mask = (np.random.uniform(size=(img_arr.shape[:2])) > 0.5).astype(np.uint8)
    random_mask = model.predict(np.expand_dims(img_arr, axis=0))[0,:,:,0]>0.5
    
    return {"prediction": random_mask.tolist()}

if __name__ == '__main__':
    app.run(debug=True)

