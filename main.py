from get_drive_model import ensure_model_download
import constants as const

def load_and_test_models():
    # Ensure the models are downloaded before starting the Flask app
    ensure_model_download(const.ANOMALY_DETECTION_MODEL_FILE_ID, 'anomaly_detection_model.h5')
    ensure_model_download(const.ANOMALY_CLASSIFICATION_MODEL_FILE_ID, 'anomaly_classification_model.h5')

