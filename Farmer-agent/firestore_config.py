# firestore_config.py
import firebase_admin
from firebase_admin import credentials, firestore

db = None

def init_firestore():
    """
    Initializes the Firestore database connection using the service account key.
    """
    global db
    if not firebase_admin._apps:
        try:
            # The path to your service account key JSON file
            cred = credentials.Certificate("serviceKey.json")
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("✅ Firestore initialized successfully.")
        except Exception as e:
            print(f"🔥 Failed to initialize Firestore: {e}")
            print("   Please ensure 'serviceAccountKey.json' is in the correct directory.")
            db = None

def get_db():
    """Returns the Firestore database client instance."""
    if db is None:
        init_firestore()
    return db