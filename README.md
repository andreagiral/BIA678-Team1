# BIA678-Team1
BIA 678 Project Contents

## Setup

1. Install PostgreSQL
2. Create database: df_db
3. Set environment variable:

DATABASE_URL=postgresql+psycopg2://postgres:password@localhost:5432/df_db

4. Install dependencies:
pip install pandas requests sqlalchemy
        optional: run installation tests
                  python 1-setup.py

6. Run ingestion:
python 2-data_ingestion.py
