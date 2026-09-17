"""
DeepShield — Auth Router (JSON file store, no JWT, no bcrypt)
POST /auth/register
POST /auth/login
"""

import hashlib
import secrets
from datetime import datetime
from fastapi import APIRouter, HTTPException, status

from backend.models.user import UserCreate, UserLogin
from backend.database import users_collection

router = APIRouter(prefix="/auth", tags=["Authentication"])


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(plain: str, stored: str) -> bool:
    try:
        salt, hashed = stored.split(":", 1)
        return hashlib.sha256((salt + plain).encode()).hexdigest() == hashed
    except Exception:
        return False


def make_token(user_id: str) -> str:
    return secrets.token_urlsafe(32) + "." + user_id


# ─── Register ────────────────────────────────────────────────────────────────
@router.post("/register", status_code=201)
async def register(user_data: UserCreate):
    col = users_collection()

    existing = col.find_one({"email": user_data.email.lower()})
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    now = datetime.utcnow()
    user_doc = {
        "name": user_data.name.strip(),
        "email": user_data.email.lower(),
        "hashed_password": hash_password(user_data.password),
        "created_at": now.isoformat(),
        "total_scans": 0,
    }
    user_id = col.insert_one(user_doc)
    token = make_token(user_id)

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "name": user_doc["name"],
            "email": user_doc["email"],
            "created_at": now.isoformat(),
            "total_scans": 0,
        },
    }


# ─── Login ───────────────────────────────────────────────────────────────────
@router.post("/login")
async def login(credentials: UserLogin):
    col = users_collection()
    user = col.find_one({"email": credentials.email.lower()})

    if not user or not verify_password(credentials.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    user_id = user["_id"]
    token = make_token(user_id)

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "name": user["name"],
            "email": user["email"],
            "created_at": user.get("created_at", ""),
            "total_scans": user.get("total_scans", 0),
        },
    }
