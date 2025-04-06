# src/SQLManager.py
import sqlite3
import pandas as pd

class DatabaseManager:
    def __init__(self, folder, db_name):
        self.conn = sqlite3.connect(f"{folder}/{db_name}", check_same_thread=False)
        self.cursor = self.conn.cursor()

    def create_recognized_plates_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognized_plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def save_recognized_plate(self, plate_text):
        self.cursor.execute("INSERT INTO recognized_plates (plate_text) VALUES (?)", (plate_text,))
        self.conn.commit()

    def get_all_recognized_plates(self):
        return pd.read_sql_query("SELECT * FROM recognized_plates ORDER BY id DESC", self.conn)
