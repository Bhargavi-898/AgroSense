# db.py
import os
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
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

