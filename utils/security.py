"""
Security helpers for encryption, backup, and input validation.
"""

import os
import re
import shutil
from datetime import datetime

import cv2
import numpy as np
from cryptography.fernet import Fernet, InvalidToken

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
INSTANCE_DIR = os.path.join(BASE_DIR, 'instance')
BACKUP_DIR = os.path.join(BASE_DIR, 'backups')
KEY_PATH = os.path.join(INSTANCE_DIR, 'data_encryption.key')
ENCRYPTED_PREFIX = 'enc::'

os.makedirs(INSTANCE_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

USERNAME_RE = re.compile(r'^[A-Za-z0-9_]{4,30}$')
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z .'-]{1,79}$")


def _get_fernet():
    env_key = os.environ.get('BRAIN_MRI_ENCRYPTION_KEY')
    if env_key:
        key = env_key.encode('utf-8')
    else:
        if not os.path.exists(KEY_PATH):
            with open(KEY_PATH, 'wb') as key_file:
                key_file.write(Fernet.generate_key())
        with open(KEY_PATH, 'rb') as key_file:
            key = key_file.read().strip()
    return Fernet(key)


def is_encrypted(value):
    return isinstance(value, str) and value.startswith(ENCRYPTED_PREFIX)


def encrypt_value(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if is_encrypted(text):
        return text
    token = _get_fernet().encrypt(text.encode('utf-8')).decode('utf-8')
    return f'{ENCRYPTED_PREFIX}{token}'


def decrypt_value(value):
    if value is None:
        return None
    if not is_encrypted(value):
        return value
    token = value[len(ENCRYPTED_PREFIX):].encode('utf-8')
    try:
        return _get_fernet().decrypt(token).decode('utf-8')
    except InvalidToken:
        return value


def encrypt_backup_bytes(data):
    return _get_fernet().encrypt(data)


def decrypt_backup_bytes(data):
    return _get_fernet().decrypt(data)


def create_encrypted_backup(db_path):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f'brain_mri_backup_{timestamp}.sqlite3.enc'
    backup_path = os.path.join(BACKUP_DIR, backup_name)
    with open(db_path, 'rb') as db_file:
        encrypted_bytes = encrypt_backup_bytes(db_file.read())
    with open(backup_path, 'wb') as backup_file:
        backup_file.write(encrypted_bytes)
    return backup_path


def restore_encrypted_backup(backup_bytes, db_path):
    decrypted_bytes = decrypt_backup_bytes(backup_bytes)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    restore_snapshot = os.path.join(BACKUP_DIR, f'pre_restore_{timestamp}.sqlite3.enc')
    if os.path.exists(db_path):
        with open(db_path, 'rb') as db_file:
            with open(restore_snapshot, 'wb') as snapshot_file:
                snapshot_file.write(encrypt_backup_bytes(db_file.read()))

    temp_path = f'{db_path}.restore_tmp'
    with open(temp_path, 'wb') as temp_file:
        temp_file.write(decrypted_bytes)
    shutil.move(temp_path, db_path)
    return restore_snapshot


def validate_username(username):
    value = (username or '').strip()
    if not USERNAME_RE.fullmatch(value):
        return 'Username must be 4-30 characters and use only letters, numbers, or underscore.'
    return None


def validate_password_strength(password):
    value = password or ''
    if len(value) < 8:
        return 'Password must be at least 8 characters long.'
    if not re.search(r'[A-Za-z]', value) or not re.search(r'\d', value):
        return 'Password must contain at least one letter and one number.'
    return None


def validate_full_name(full_name):
    value = (full_name or '').strip()
    if not NAME_RE.fullmatch(value):
        return 'Full name should be 2-80 characters and contain only letters, spaces, apostrophes, dots, or hyphens.'
    return None


def validate_age(age):
    if age in (None, ''):
        return None
    try:
        age_value = int(age)
    except (TypeError, ValueError):
        return 'Age must be a valid number.'
    if not 1 <= age_value <= 120:
        return 'Age must be between 1 and 120.'
    return None


def validate_gender(gender):
    if (gender or 'Unknown') not in {'Male', 'Female', 'Other', 'Unknown'}:
        return 'Invalid gender selected.'
    return None


def validate_role(role):
    if (role or '').lower() not in {'admin', 'doctor', 'patient'}:
        return 'Invalid role selected.'
    return None


def validate_image_bytes(file_bytes):
    if not file_bytes:
        return 'Uploaded file is empty.'
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 'Uploaded file is not a valid MRI image.'
    return None
