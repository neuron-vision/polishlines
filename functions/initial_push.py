import json
from pathlib import Path
from datetime import timedelta
import firebase_admin
from firebase_admin import credentials, db, storage
from tqdm import tqdm
from os.path import abspath

FUNCTIONS_DIR = Path(abspath(__file__)).parent
ROOT_DIR = FUNCTIONS_DIR.parent
DATA_FOLDER = ROOT_DIR / "scan/PLScanDB/"
SERVICE_ACCOUNT = list(FUNCTIONS_DIR.glob("service_accounts/*.json"))[0]
print(f"Using service account: {SERVICE_ACCOUNT}")

cred = credentials.Certificate(str(SERVICE_ACCOUNT))
firebase_admin.initialize_app(cred, {
    'storageBucket': 'polish-lines.firebasestorage.app',
    'databaseURL': 'https://polish-lines-default-rtdb.firebaseio.com'
})
ref = db.reference('folders')
bucket = storage.bucket()

existing = ref.get(shallow=True) or {}
print(f"Found {len(existing)} existing folders in DB")

all_folders = [f for f in DATA_FOLDER.iterdir() if f.is_dir()]

for folder in tqdm(sorted(all_folders)):
    if not folder.is_dir():
        continue
    folder_name = folder.name
    if folder_name in existing:
        print(f"Skipping {folder_name} (exists)")
        continue
    print(f"Processing {folder_name}")
    
    extra_data_path = folder / "Extra Data.json"
    contour_path = folder / "Contour.json"
    extra_data = json.load(open(extra_data_path)) if extra_data_path.exists() else {}
    contour_data = json.load(open(contour_path)) if contour_path.exists() else {}
    
    download_links = {}
    for png_file in folder.glob("*.png"):
        blob = bucket.blob(f"original_images/{folder_name}/{png_file.name}")
        blob.upload_from_filename(str(png_file))
        url = blob.generate_signed_url(expiration=timedelta(days=7))
        download_links[png_file.stem] = url
        print(f"  Uploaded {png_file.name}")
    
    label = extra_data.get("Polish Lines Data_User Input", extra_data.get("Polish Lines Data", {}).get("User Input", "Unknown"))

    doc_data = {
        **extra_data.get("Polish Lines Data", {}),
        **contour_data.get("Contour Data", {}),
        "download_links": download_links,
        "label": label,
        "angles": extra_data.get("Polish Lines Data", {}).get("Chosen Facet PD", None)
    }
    ref.child(folder_name).set(doc_data)
    print(f"  RTDB updated")

print("Done")
