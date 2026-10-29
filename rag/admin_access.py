"""Separate administrator credential; anonymous visitors never receive it."""
import hmac
import os


def admin_password():
    return os.getenv('ADMIN_PASSWORD', '')


def valid_admin_password(value):
    expected = admin_password()
    return bool(expected and isinstance(value, str) and
                hmac.compare_digest(value.encode(), expected.encode()))
