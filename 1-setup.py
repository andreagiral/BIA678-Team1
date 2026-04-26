try:
    import pandas
    print("Pandas is installed")
except ImportError:
    print("Pandas is not installed")

try:
    import requests
    print("Requests is installed")
except ImportError:
    print("Requests is not installed")

try:
    import sqlalchemy
    print("SQLAlchemy is installed")
except ImportError:
    print("SQLAlchemy is not installed")