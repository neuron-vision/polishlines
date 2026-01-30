import json
from pathlib import Path
from datetime import timedelta, datetime
import firebase_admin
from firebase_admin import credentials, db, storage
from os.path import abspath

def push_model():
    FUNCTIONS_DIR = Path(abspath(__file__)).parent
    ROOT_DIR = FUNCTIONS_DIR.parent
    MODEL_PATH = ROOT_DIR / "hrnet_w48_int8.onnx"
    EMBEDDER_NAME = "hrnet_w48"
    
    # Discovery of service account
    service_accounts = list(FUNCTIONS_DIR.glob("service_accounts/*.json"))
    if not service_accounts:
        print("No service account found in functions/service_accounts/")
        return
    
    SERVICE_ACCOUNT = service_accounts[0]
    print(f"Using service account: {SERVICE_ACCOUNT}")

    cred = credentials.Certificate(str(SERVICE_ACCOUNT))
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'polish-lines.firebasestorage.app',
        'databaseURL': 'https://polish-lines-default-rtdb.firebaseio.com'
    })

    bucket = storage.bucket()
    storage_path = f"embedder_models/{EMBEDDER_NAME}/int8.onnx"
    blob = bucket.blob(storage_path)

    print(f"Uploading {MODEL_PATH} to {storage_path}...")
    blob.upload_from_filename(str(MODEL_PATH))

    # Expiration in 1 year
    expiration = timedelta(days=365)
    download_url = blob.generate_signed_url(expiration=expiration)
    print(f"Generated download link: {download_url}")

    timestamp = datetime.now().isoformat()
    meta_data = {
        "url": download_url,
        "timestamp": timestamp,
        "name": EMBEDDER_NAME
    }

    print("Updating Realtime Database...")
    db.reference('metadata/embedders/latest').set(meta_data)
    db.reference(f'metadata/embedders/{EMBEDDER_NAME}').set(meta_data)

    print("Successfully pushed embedder model and updated metadata.")

if __name__ == "__main__":
    push_model()
