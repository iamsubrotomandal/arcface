#!/usr/bin/env python3
"""
Database utilities for student registration system.
Handles SQLite database operations for storing student information.
"""

import sqlite3
import os
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

class StudentDatabase:
    """Database manager for student registration system."""
    
    def __init__(self, db_path: str = "data/students.db"):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Create the student table if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create student table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS student (
                        student_id TEXT PRIMARY KEY,
                        first_name TEXT NOT NULL,
                        middle_name TEXT,
                        last_name TEXT NOT NULL,
                        date_of_birth TEXT NOT NULL,
                        gender TEXT NOT NULL,
                        phone TEXT NOT NULL,
                        email TEXT NOT NULL,
                        program TEXT NOT NULL,
                        academic_year TEXT,
                        current_semester TEXT,
                        emergency_contact TEXT,
                        guardian_name TEXT,
                        address TEXT,
                        notes TEXT,
                        registration_date TEXT NOT NULL,
                        face_embedding_path TEXT,
                        is_active BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create index on email for faster lookups
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_student_email 
                    ON student(email)
                ''')
                
                conn.commit()
                logging.info("Database initialized successfully")
                
        except sqlite3.Error as e:
            logging.error(f"Database initialization error: {e}")
            raise
    
    def insert_student(self, student_data: Dict[str, Any]) -> bool:
        """Insert a new student record."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if student ID already exists
                cursor.execute("SELECT student_id FROM student WHERE student_id = ?", 
                             (student_data['student_id'],))
                if cursor.fetchone():
                    logging.warning(f"Student ID {student_data['student_id']} already exists")
                    return False
                
                # Insert new student
                cursor.execute('''
                    INSERT INTO student (
                        student_id, first_name, middle_name, last_name, 
                        date_of_birth, gender, phone, email, program,
                        academic_year, current_semester, emergency_contact,
                        guardian_name, address, notes, registration_date,
                        face_embedding_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    student_data['student_id'],
                    student_data['first_name'],
                    student_data.get('middle_name', ''),
                    student_data['last_name'],
                    student_data['date_of_birth'],
                    student_data['gender'],
                    student_data['phone'],
                    student_data['email'],
                    student_data['program'],
                    student_data.get('year', ''),
                    student_data.get('semester', ''),
                    student_data.get('emergency_contact', ''),
                    student_data.get('guardian_name', ''),
                    student_data.get('address', ''),
                    student_data.get('notes', ''),
                    student_data['registration_date'],
                    student_data.get('face_embedding_path', '')
                ))
                
                conn.commit()
                logging.info(f"Student {student_data['student_id']} registered successfully")
                return True
                
        except sqlite3.Error as e:
            logging.error(f"Error inserting student: {e}")
            return False
    
    def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a student record by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Enable dict-like access
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM student WHERE student_id = ?", (student_id,))
                row = cursor.fetchone()
                
                if row:
                    return dict(row)
                return None
                
        except sqlite3.Error as e:
            logging.error(f"Error retrieving student: {e}")
            return None
    
    def get_all_students(self) -> List[Dict[str, Any]]:
        """Retrieve all active students."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM student 
                    WHERE is_active = 1 
                    ORDER BY registration_date DESC
                """)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except sqlite3.Error as e:
            logging.error(f"Error retrieving students: {e}")
            return []
    
    def update_student(self, student_id: str, update_data: Dict[str, Any]) -> bool:
        """Update a student record."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build dynamic update query
                fields = []
                values = []
                for key, value in update_data.items():
                    if key != 'student_id':  # Don't update primary key
                        fields.append(f"{key} = ?")
                        values.append(value)
                
                if not fields:
                    return False
                
                # Add updated timestamp
                fields.append("updated_at = CURRENT_TIMESTAMP")
                values.append(student_id)
                
                query = f"UPDATE student SET {', '.join(fields)} WHERE student_id = ?"
                cursor.execute(query, values)
                
                conn.commit()
                return cursor.rowcount > 0
                
        except sqlite3.Error as e:
            logging.error(f"Error updating student: {e}")
            return False
    
    def delete_student(self, student_id: str) -> bool:
        """Soft delete a student (mark as inactive)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE student 
                    SET is_active = 0, updated_at = CURRENT_TIMESTAMP 
                    WHERE student_id = ?
                """, (student_id,))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except sqlite3.Error as e:
            logging.error(f"Error deleting student: {e}")
            return False
    
    def get_student_count(self) -> int:
        """Get total number of active students."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM student WHERE is_active = 1")
                count = cursor.fetchone()[0]
                return count
                
        except sqlite3.Error as e:
            logging.error(f"Error getting student count: {e}")
            return 0
    
    def search_students(self, search_term: str) -> List[Dict[str, Any]]:
        """Search students by name, ID, or email."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                search_pattern = f"%{search_term}%"
                cursor.execute("""
                    SELECT * FROM student 
                    WHERE is_active = 1 AND (
                        student_id LIKE ? OR 
                        first_name LIKE ? OR 
                        last_name LIKE ? OR 
                        email LIKE ?
                    )
                    ORDER BY last_name, first_name
                """, (search_pattern, search_pattern, search_pattern, search_pattern))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except sqlite3.Error as e:
            logging.error(f"Error searching students: {e}")
            return []
    
    def check_email_exists(self, email: str, exclude_student_id: Optional[str] = None) -> bool:
        """Check if email already exists for another student."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if exclude_student_id:
                    cursor.execute("""
                        SELECT COUNT(*) FROM student 
                        WHERE email = ? AND student_id != ? AND is_active = 1
                    """, (email, exclude_student_id))
                else:
                    cursor.execute("""
                        SELECT COUNT(*) FROM student 
                        WHERE email = ? AND is_active = 1
                    """, (email,))
                
                count = cursor.fetchone()[0]
                return count > 0
                
        except sqlite3.Error as e:
            logging.error(f"Error checking email: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total students
                cursor.execute("SELECT COUNT(*) FROM student WHERE is_active = 1")
                total_students = cursor.fetchone()[0]
                
                # Students by program
                cursor.execute("""
                    SELECT program, COUNT(*) 
                    FROM student 
                    WHERE is_active = 1 
                    GROUP BY program
                """)
                programs = dict(cursor.fetchall())
                
                # Recent registrations (last 7 days)
                cursor.execute("""
                    SELECT COUNT(*) FROM student 
                    WHERE is_active = 1 
                    AND date(registration_date) >= date('now', '-7 days')
                """)
                recent_registrations = cursor.fetchone()[0]
                
                return {
                    'total_students': total_students,
                    'programs': programs,
                    'recent_registrations': recent_registrations
                }
                
        except sqlite3.Error as e:
            logging.error(f"Error getting database stats: {e}")
            return {
                'total_students': 0,
                'programs': {},
                'recent_registrations': 0
            }
