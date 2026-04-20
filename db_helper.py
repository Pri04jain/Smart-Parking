# db_helper.py
import mysql.connector
from mysql.connector import Error
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DatabaseHelper:
    def __init__(self):
        self.connection = None
        self.connect()

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host='localhost',
                user='root',
                password='',
                database='db1',
                autocommit=False
            )
            logging.info("Connected to MySQL database")
        except Error as e:
            logging.error(f"Error connecting to MySQL: {e}")
            raise

    def update_parking_status(self, available, occupied, location="default", image_path=None):
        """
        Updates parking status with all required fields
        Args:
            available: Number of available spaces
            occupied: Number of occupied spaces
            location: Parking location identifier
            image_path: Path to the processed image (optional)
        """
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()

            cursor = self.connection.cursor()
            total_spaces = available + occupied
            
            query = """
            INSERT INTO parking_status 
            (location, available_spaces, occupied_spaces, total_spaces, image_path, last_updated)
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON DUPLICATE KEY UPDATE
                available_spaces = VALUES(available_spaces),
                occupied_spaces = VALUES(occupied_spaces),
                total_spaces = VALUES(total_spaces),
                image_path = VALUES(image_path),
                last_updated = NOW()
            """
            
            cursor.execute(query, (
                location,
                available,
                occupied,
                total_spaces,
                str(image_path) if image_path else None
            ))
            
            self.connection.commit()
            logging.info(
                f"Updated parking status - Available: {available}, "
                f"Occupied: {occupied}, "
                f"Total: {total_spaces}, "
                f"Image: {image_path}"
            )
            return True
            
        except Error as e:
            self.connection.rollback()
            logging.error(f"Database error: {e}")
            return False
        finally:
            if 'cursor' in locals() and cursor:
                cursor.close()

    def get_latest_status(self, location="default"):
        """
        Retrieves the latest parking status for a location
        Returns: Dictionary with status data or None if error
        """
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()

            cursor = self.connection.cursor(dictionary=True)
            
            query = """
            SELECT 
                available_spaces, 
                occupied_spaces, 
                total_spaces,
                image_path,
                last_updated
            FROM parking_status
            WHERE location = %s
            ORDER BY last_updated DESC
            LIMIT 1
            """
            
            cursor.execute(query, (location,))
            result = cursor.fetchone()
            
            return result if result else None
            
        except Error as e:
            logging.error(f"Error fetching parking status: {e}")
            return None
        finally:
            if 'cursor' in locals() and cursor:
                cursor.close()

    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logging.info("MySQL connection closed")

# Singleton instance
db_helper = DatabaseHelper()