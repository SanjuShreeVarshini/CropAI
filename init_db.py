import sqlite3

conn = sqlite3.connect("database.db")

# USERS TABLE
conn.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    role TEXT
)
""")

# CROP SELL REQUEST TABLE
conn.execute("""
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    farmer TEXT,
    crop TEXT,
    quantity INTEGER,
    status TEXT
)
""")

conn.commit()
conn.close()

print("Database Created Successfully")