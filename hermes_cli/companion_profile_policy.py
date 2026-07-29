import hashlib
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import yaml

try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import msvcrt
except ImportError:
    msvcrt = None

LEGACY_POLICY = "legacy"
GUARDED_V1_POLICY = "guarded_v1"
GUARDED_V2_POLICY = "guarded_v2"
STYLE_GUARD_V1_POLICY = "style_guard_v1"
GUARDED_POLICY = GUARDED_V1_POLICY
PROFILE_META_FILE = "profile.yaml"


def _read_profile_payload(profile_dir):
    path = Path(profile_dir) / PROFILE_META_FILE
    if not path.is_file():
        return None
    try:
        with open(str(path), "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def read_conversation_policy(profile_dir):
    payload = _read_profile_payload(profile_dir)
    if payload is None:
        return STYLE_GUARD_V1_POLICY
    if payload.get("conversation_policy") == LEGACY_POLICY:
        return LEGACY_POLICY
    return STYLE_GUARD_V1_POLICY


def read_evolution_policy(profile_dir):
    payload = _read_profile_payload(profile_dir)
    if payload is None:
        return GUARDED_V2_POLICY
    evolution_policy = payload.get("evolution_policy")
    if evolution_policy in (GUARDED_V1_POLICY, GUARDED_V2_POLICY):
        return evolution_policy
    return GUARDED_V2_POLICY


def _atomic_write_yaml(path, payload):
    path = Path(path)
    temp_path = path.with_name(".{0}.{1}.tmp".format(path.name, uuid.uuid4().hex))
    try:
        with open(str(temp_path), "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temp_path), str(path))
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _profile_init_lock(profile_dir):
    profile_dir = Path(profile_dir)
    lock_dir = profile_dir.parent / ".init-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(str(profile_dir).encode("utf-8")).hexdigest() + ".lock"
    lock_path = lock_dir / lock_name
    with open(str(lock_path), "a+b") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def profile_lock(profile_dir, purpose):
    profile_dir = Path(profile_dir)
    lock_dir = profile_dir.parent / ".profile-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_key = "{0}:{1}".format(str(profile_dir), str(purpose))
    lock_name = hashlib.sha256(lock_key.encode("utf-8")).hexdigest() + ".lock"
    lock_path = lock_dir / lock_name
    with open(str(lock_path), "a+b") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def ensure_companion_profile(profile_dir):
    profile_dir = Path(profile_dir)
    profile_dir.parent.mkdir(parents=True, exist_ok=True)
    with _profile_init_lock(profile_dir):
        if profile_dir.exists():
            if not profile_dir.is_dir():
                raise IOError("companion profile path is not a directory: {0}".format(profile_dir))
            meta_path = profile_dir / PROFILE_META_FILE
            if not meta_path.is_file():
                _atomic_write_yaml(
                    meta_path,
                    {
                        "conversation_policy": STYLE_GUARD_V1_POLICY,
                        "evolution_policy": GUARDED_V2_POLICY,
                    },
                )
            return read_evolution_policy(profile_dir)

        profile_dir.mkdir()
        try:
            _atomic_write_yaml(
                profile_dir / PROFILE_META_FILE,
                {
                    "conversation_policy": STYLE_GUARD_V1_POLICY,
                    "evolution_policy": GUARDED_V2_POLICY,
                },
            )
        except Exception:
            shutil.rmtree(str(profile_dir), ignore_errors=True)
            raise
        return GUARDED_V2_POLICY
