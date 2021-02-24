from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np


def get_model_scores(face1, face2):
    faces = []
    face1 = img_to_array(face1)
    face1 = preprocess_input(face1)
    face2 = img_to_array(face2)
    face2 = preprocess_input(face2)
    faces.append(face2)
    model = VGGFace(model='resnet50',
                    input_shape=(224, 224, 3),
                    pooling='avg')

    return model.predict(faces)
