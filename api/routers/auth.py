from fastapi import APIRouter, Depends, Request, Response, status

from api.auth import SESSION_COOKIE, AuthRepository, authenticate, create_session, register_user
from api.schemas.auth import LoginRequest, RegisterRequest, UserView

router = APIRouter(prefix="/auth", tags=["authentication"])


def get_auth_repository(request: Request) -> AuthRepository:
    return request.app.state.auth_repository


@router.post("/register", response_model=UserView, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, repository: AuthRepository = Depends(get_auth_repository)):
    user = register_user(repository, payload.username, payload.password)
    return UserView(id=user.id, username=user.username)


@router.post("/login", response_model=UserView)
def login(payload: LoginRequest, response: Response, repository: AuthRepository = Depends(get_auth_repository)):
    user = authenticate(repository, payload.username, payload.password)
    response.set_cookie(SESSION_COOKIE, create_session(repository, user.id), httponly=True, samesite="lax")
    return UserView(id=user.id, username=user.username)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request, response: Response, repository: AuthRepository = Depends(get_auth_repository)):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        repository.delete_session(token)
    response.delete_cookie(SESSION_COOKIE)
