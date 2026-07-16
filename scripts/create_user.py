"""Create or update a user account.

    .venv\\Scripts\\python.exe scripts\\create_user.py alice s3cret developer
Roles: admin | developer | viewer   (see ROLES in app/config.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import auth, db                     # noqa: E402
from app.config import ROLES                 # noqa: E402


def main():
    if len(sys.argv) != 4 or sys.argv[3] not in ROLES:
        print(__doc__)
        sys.exit(1)
    username, password, role = sys.argv[1:4]
    db.get_conn()
    db.create_user(username, auth.hash_password(password), role)
    print(f"user '{username}' saved with role '{role}'")


if __name__ == "__main__":
    main()
