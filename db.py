# db.py
import os
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = "mongodb+srv://bhargavi10112005:Cyrus2005@cluster0.upknunx.mongodb.net/?appName=Cluster0"
DB_NAME = os.getenv("DB_NAME", "argosense_db")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set in .env")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

users_col = db["users"]
history_col = db["history"]

def create_user(email: str, password_hash: str):
    """Return (ok: bool, msg: str)."""
    if users_col.find_one({"email": email}):
        return False, "Email already registered"
    users_col.insert_one({
        "email": email,
        "password": password_hash,
        "created_at": datetime.utcnow()
    })
    return True, "User created"

def get_user_by_email(email: str):
    return users_col.find_one({"email": email})
def update_password(email: str, new_password_hash: bytes):
    user = users_col.find_one({"email": email})
    if not user:
        return False, "Email not found"
    users_col.update_one(
        {"email": email},
        {"$set": {"password": new_password_hash}}
    )
    return True, "Password updated successfully"

# ================== HISTORY FUNCTIONS ==================

def save_history(user_email: str, location: str, input_data: dict, result_data: dict):
    """
    Save a single crop prediction for a user at a given location
    """
    history_col.insert_one({
        "user": st.session_state['current_user'],
        "city": city,
        "month": datetime.datetime.now().month,
        "N": st.session_state['N'],
        "P": st.session_state['P'],
        "K": st.session_state['K'],
        "pH": st.session_state['ph'],
        "temperature": temp,
        "humidity": hum,
        "rainfall": rain,
        "recommendation": st.session_state.get("recommendation"),
        "timestamp": datetime.utcnow()
})



def get_history(user_email: str, location: str, limit: int = 10):
    """
    Get last 'limit' history records for a user at a location
    """
    cursor = history_col.find({
        "user_email": user_email,
        "location": location.lower()
    }).sort("timestamp", -1).limit(limit)
    return list(cursor)


def get_last_history(user_email: str, location: str):
    """
    Get only the most recent history for a user at a location
    """
    records = get_history(user_email, location, limit=1)
    return records[0] if records else None
