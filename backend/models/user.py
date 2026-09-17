"""
DeepShield — User Pydantic Models (No JWT)
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="Full name")
    email: str
    password: str = Field(..., min_length=6, description="Minimum 6 characters")


class UserLogin(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: str
    name: str
    email: str
    created_at: datetime
    total_scans: int = 0
