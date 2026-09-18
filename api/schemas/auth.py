from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3)
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    username: str
    password: str


class UserView(BaseModel):
    id: str
    username: str
