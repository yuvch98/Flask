from get_drive_model import ensure_model_download
import constants as const
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub


def load_and_test_models():
    # Ensure the models are downloaded before starting the Flask app
    # ensure_model_download(const.ANOMALY_DETECTION_MODEL_FILE_ID, 'anomaly_detection_model.h5')
    # ensure_model_download(const.ANOMALY_CLASSIFICATION_MODEL_FILE_ID, 'anomaly_classification_model.h5')
    print(tf.__version__)
    print(hub.__version__)
    #
    # # Define a simple model with the TensorFlow Hub layer to test
    input_shape = (50, 224, 224, 3)  # Example shape, adjust as necessary
    inputs = tf.keras.Input(shape=input_shape)

    custom_objects = {'KerasLayer' : hub.KerasLayer}
    try:
        detection_model = tf.keras.models.load_model('anomaly_detection_model.h5', custom_objects=custom_objects)
        classification_model = tf.keras.models.load_model('anomaly_classification_model.h5',
                                                          custom_objects=custom_objects)
    except Exception as e:
        print("Error loading models with custom scope:", e)
        return None, None

    return detection_model, classification_model

if __name__ == '__main__':
    anomaly_detection_model, anomaly_classification_model = load_and_test_models()
    print(anomaly_detection_model)
