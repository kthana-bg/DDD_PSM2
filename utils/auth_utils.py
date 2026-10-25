import hashlib
import hmac
import os
import re

from database.db_manager import create_user, get_user_by_username

PBKDF2_ITERATIONS = 200_000


def hash_password(password: str, salt: bytes = None) -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2${PBKDF2_ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iters, salt_hex, hash_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


def register_user(full_name: str, username: str, password: str) -> dict:
    full_name, username = (full_name or "").strip(), (username or "").strip().lower()
    if not full_name:
        return {"success": False, "message": "Please enter your full name."}
    if not re.fullmatch(r"[a-z0-9_.-]{3,30}", username):
        return {"success": False, "message": "Username must be 3-30 characters: letters, numbers, . _ -"}
    if len(password or "") < 6:
        return {"success": False, "message": "Password must be at least 6 characters."}
    if get_user_by_username(username):
        return {"success": False, "message": "That username is already taken."}
    uid = create_user(username, hash_password(password), full_name)
    if uid == -1:
        return {"success": False, "message": "Could not create the account. Please try again."}
    return {"success": True, "message": "Account created. You can now sign in.", "user_id": uid}


def login_user(username: str, password: str) -> dict:
    user = get_user_by_username((username or "").strip().lower())
    if not user or not verify_password(password or "", user["password_hash"]):
        return {"success": False, "message": "Wrong username or password."}
    user = {k: v for k, v in user.items() if k != "password_hash"}
    return {"success": True, "message": f"Welcome back, {user.get('full_name') or user['username']}.", "user": user}
