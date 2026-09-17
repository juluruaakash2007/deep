"""
DeepShield — Auth Dependencies (JSON store, no JWT)
Extracts user_id from Bearer token and looks up in JSON store.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from backend.database import users_collection

bearer_scheme = HTTPBearer(auto_error=False)


def _user_id_from_token(token: str) -> str | None:
    try:
        return token.split(".")[-1]
    except Exception:
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = _user_id_from_token(credentials.credentials)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token.")

    col = users_collection()
    # JSON store keeps users keyed by user_id
    from backend.database import _load
    data = _load()
    user = data["users"].get(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")

    user["id"] = user_id
    user["_id"] = user_id
    return user
