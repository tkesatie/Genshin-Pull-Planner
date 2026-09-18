"""User authentication and persistent sessions for the prototype."""

import hashlib
import hmac
import secrets
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4


class AuthenticationError(ValueError):
    """Raised when credentials are invalid or a user already exists."""


@dataclass(frozen=True)
class UserRecord:
    id: str
    username: str
    password_hash: str


class AuthRepository:
    def create_user(self, user: UserRecord) -> UserRecord: raise NotImplementedError
    def get_user_by_username(self, username: str) -> UserRecord | None: raise NotImplementedError
    def create_session(self, token: str, user_id: str, expires_at: datetime) -> None: raise NotImplementedError
    def get_session(self, token: str) -> tuple[str, datetime] | None: raise NotImplementedError
    def delete_session(self, token: str) -> None: raise NotImplementedError


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return "scrypt$16384$8$1$" + salt.hex() + "$" + digest.hex()


def _verify_password(password: str, encoded: str) -> bool:
    try:
        _, n, r, p, salt_hex, digest_hex = encoded.split("$")
        digest = hashlib.scrypt(
            password.encode(),
            salt=bytes.fromhex(salt_hex),
            n=int(n),
            r=int(r),
            p=int(p),
        )
        return hmac.compare_digest(digest, bytes.fromhex(digest_hex))
    except (ValueError, TypeError):
        return False


class InMemoryAuthRepository(AuthRepository):
    def __init__(self):
        self._users = {}
        self._sessions = {}
        self._lock = threading.Lock()

    def create_user(self, user):
        with self._lock:
            if any(existing.username == user.username for existing in self._users.values()):
                raise AuthenticationError("username already exists")
            self._users[user.id] = user
        return user

    def get_user_by_username(self, username):
        with self._lock:
            return next((u for u in self._users.values() if u.username == username), None)

    def create_session(self, token, user_id, expires_at):
        with self._lock:
            self._sessions[token] = (user_id, expires_at)

    def get_session(self, token):
        with self._lock:
            return self._sessions.get(token)

    def delete_session(self, token):
        with self._lock:
            self._sessions.pop(token, None)


class SQLiteAuthRepository(AuthRepository):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )""")

    def _connect(self):
        return sqlite3.connect(self.path)

    def create_user(self, user):
        with self._lock, self._connect() as db:
            try:
                db.execute(
                    "INSERT INTO users (id, username, password_hash) VALUES (?, ?, ?)",
                    (user.id, user.username, user.password_hash),
                )
            except sqlite3.IntegrityError as exc:
                raise AuthenticationError("username already exists") from exc
        return user

    def get_user_by_username(self, username):
        with self._lock, self._connect() as db:
            row = db.execute(
                "SELECT id, username, password_hash FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        return None if row is None else UserRecord(*row)

    def create_session(self, token, user_id, expires_at):
        with self._lock, self._connect() as db:
            db.execute(
                "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
                (token, user_id, expires_at.isoformat()),
            )

    def get_session(self, token):
        with self._lock, self._connect() as db:
            row = db.execute(
                "SELECT user_id, expires_at FROM sessions WHERE token = ?",
                (token,),
            ).fetchone()
        if row is None:
            return None
        return row[0], datetime.fromisoformat(row[1])

    def delete_session(self, token):
        with self._lock, self._connect() as db:
            db.execute("DELETE FROM sessions WHERE token = ?", (token,))


SESSION_COOKIE = "genshin_session"
SESSION_DURATION = timedelta(days=30)


def register_user(repository, username: str, password: str) -> UserRecord:
    username = username.strip()
    if len(username) < 3:
        raise AuthenticationError("username must be at least 3 characters")
    if len(password) < 8:
        raise AuthenticationError("password must be at least 8 characters")
    return repository.create_user(UserRecord(uuid4().hex, username, _hash_password(password)))


def authenticate(repository, username: str, password: str) -> UserRecord:
    user = repository.get_user_by_username(username.strip())
    if user is None or not _verify_password(password, user.password_hash):
        raise AuthenticationError("invalid username or password")
    return user


def create_session(repository, user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    repository.create_session(token, user_id, datetime.now(timezone.utc) + SESSION_DURATION)
    return token
