from app import db  # Assuming 'myapp' is your Flask application instance
from backend.database.models import User
from sqlalchemy.exc import IntegrityError

def register_user(username, password):
    if not username or not password:
        return False

    try:
        # Create a new User object
        new_user = User(username=username, password=password)

        # Add new_user to the database session
        db.session.add(new_user)

        # Commit changes to the database
        db.session.commit()

        return True
    except IntegrityError:
        # Handle IntegrityError (e.g., duplicate username)
        db.session.rollback()
        return False
    except Exception as e:
        # Handle other exceptions
        db.session.rollback()
        print(f"Error: {e}")
        return False
