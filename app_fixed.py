#!/usr/bin/env python3
"""
Face Recognition System - Student Registration Interface

A simple Streamlit-based frontend for student registration with face capture,
recognition, and database management.
"""

import streamlit as st
import cv2
import numpy as np
import os
import sys
from PIL import Image
import tempfile
import time
from typing import Optional, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import backend components
try:
    from pipelines.still_image_pipeline import StillImageFacePipeline
    from utils.face_db import FaceDB
    BACKEND_AVAILABLE = True
except ImportError as e:
    st.error(f"Backend components not available: {e}")
    BACKEND_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="ArcFace Student Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'face_pipeline' not in st.session_state:
    st.session_state.face_pipeline = None
if 'face_db' not in st.session_state:
    st.session_state.face_db = None
if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None
if 'student_data' not in st.session_state:
    st.session_state.student_data = {}
if 'registration_step' not in st.session_state:
    st.session_state.registration_step = 1
if 'current_page' not in st.session_state:
    st.session_state.current_page = "register_student"

def get_device():
    """Get the appropriate device for processing."""
    device_options = ["cpu", "cuda"]
    return st.sidebar.selectbox("Processing Device", device_options, index=0)

def initialize_system() -> bool:
    """Initialize the face recognition system."""
    if not BACKEND_AVAILABLE:
        st.error("❌ Backend components are not available. Please check your installation.")
        return False
    
    if st.session_state.face_pipeline is not None:
        return True
    
    try:
        with st.spinner("🔧 Initializing face recognition system..."):
            device = get_device()
            
            # Initialize face database
            db_path = "data/face_database.pkl"
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            st.session_state.face_db = FaceDB(db_path)
            
            # Initialize face pipeline
            st.session_state.face_pipeline = StillImageFacePipeline(
                device=device,
                face_db=st.session_state.face_db
            )
            
            st.success("✅ System initialized successfully!")
            return True
            
    except Exception as e:
        st.error(f"❌ Failed to initialize system: {str(e)}")
        return False

def show_database_stats():
    """Display database statistics in sidebar."""
    if st.session_state.face_db is not None:
        try:
            num_faces = st.session_state.face_db.count()
            st.metric("Registered Students", num_faces)
            
            if num_faces > 0:
                with st.expander("View Registered Students"):
                    for identity in st.session_state.face_db.ids:
                        st.text(f"• {identity}")
        except Exception as e:
            st.error(f"Error reading database: {str(e)}")
    else:
        st.metric("Registered Students", 0)

def capture_photo_from_upload():
    """Handle photo upload for registration."""
    st.markdown('<div class="section-header">📸 Upload Student Photo</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a photo...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear photo of the student's face"
    )
    
    if uploaded_file is not None:
        # Convert uploaded file to image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_array.shape) == 3:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_array
        
        # Store the captured image
        st.session_state.captured_image = image_bgr
        
        # Process the face
        with st.spinner("🔍 Processing face..."):
            try:
                result = st.session_state.face_pipeline.process_image(image_bgr)
                
                if result['faces_detected']:
                    st.session_state.face_data = result['results'][0]
                    st.success("✅ Face detected and processed successfully!")
                    st.session_state.registration_step = 2
                    st.rerun()
                else:
                    st.error("❌ No face detected in the image. Please upload a clear photo with a visible face.")
                    
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
    
    # Show captured image if available
    if st.session_state.captured_image is not None:
        st.subheader("📷 Captured Image")
        
        # Create a copy for display
        if 'face_data' in st.session_state and 'bbox' in st.session_state.face_data:
            display_image = st.session_state.captured_image.copy()
            bbox = st.session_state.face_data['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Convert BGR to RGB for display
            display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
            st.image(display_image_rgb, caption="Detected Face", use_column_width=True)

def student_registration_form():
    """Display student registration form."""
    st.subheader("👤 Student Registration Form")
    
    # Show face analysis results
    if 'face_data' in st.session_state:
        face_data = st.session_state.face_data
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Liveness Score", f"{face_data['liveness_score']:.3f}")
        with col2:
            st.metric("Face Quality", "✅ Good" if face_data['liveness_score'] > 0.7 else "⚠️ Fair")
    
    # Registration form
    with st.form("student_registration"):
        col1, col2 = st.columns(2)
        
        with col1:
            student_id = st.text_input("Student ID *", placeholder="e.g., 2024001")
            first_name = st.text_input("First Name *", placeholder="e.g., John")
            last_name = st.text_input("Last Name *", placeholder="e.g., Doe")
            email = st.text_input("Email *", placeholder="e.g., john.doe@university.edu")
        
        with col2:
            department = st.selectbox("Department *", [
                "Computer Science",
                "Engineering", 
                "Mathematics",
                "Physics",
                "Chemistry",
                "Biology",
                "Business",
                "Psychology",
                "Other"
            ])
            year = st.selectbox("Year *", ["1st Year", "2nd Year", "3rd Year", "4th Year", "Graduate"])
            phone = st.text_input("Phone Number", placeholder="e.g., +1234567890")
            emergency_contact = st.text_input("Emergency Contact", placeholder="e.g., +1234567890")
        
        notes = st.text_area("Additional Notes", placeholder="Any additional information...")
        
        # Submit button
        submit_button = st.form_submit_button("🎓 Register Student", type="primary")
        
        if submit_button:
            # Validate required fields
            if not all([student_id, first_name, last_name, email, department, year]):
                st.error("❌ Please fill in all required fields marked with *")
            else:
                # Prepare student data
                student_data = {
                    'student_id': student_id,
                    'first_name': first_name,
                    'last_name': last_name,
                    'email': email,
                    'department': department,
                    'year': year,
                    'phone': phone,
                    'emergency_contact': emergency_contact,
                    'notes': notes,
                    'registration_date': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Register student
                if register_student(student_data):
                    st.success("🎉 Student registered successfully!")
                    st.balloons()
                    
                    # Reset for next registration
                    time.sleep(2)
                    st.session_state.registration_step = 1
                    st.session_state.captured_image = None
                    st.session_state.student_data = {}
                    if 'face_data' in st.session_state:
                        del st.session_state.face_data
                    st.rerun()

def register_student(student_data: Dict[str, str]) -> bool:
    """Register student in the face database."""
    try:
        if 'face_data' not in st.session_state:
            st.error("No face data available")
            return False
        
        # Create identity string
        identity = f"{student_data['student_id']} - {student_data['first_name']} {student_data['last_name']}"
        
        # Add to face database using the correct API
        embedding = st.session_state.face_data['embedding']
        
        # Convert to numpy array if it's a tensor
        if hasattr(embedding, 'cpu'):
            embedding_np = embedding.cpu().numpy()
        elif hasattr(embedding, 'numpy'):
            embedding_np = embedding.numpy()
        else:
            embedding_np = np.array(embedding)
        
        # Ensure it's 1D
        if embedding_np.ndim > 1:
            embedding_np = embedding_np.flatten()
        
        # Enroll in face database
        st.session_state.face_db.enroll(identity, embedding_np)
        
        return True
            
    except Exception as e:
        st.error(f"Registration failed: {str(e)}")
        return False

def show_register_student_page():
    """Show the Register New Student page."""
    st.header("👤 Register New Student")
    st.markdown("Upload a student photo and fill in their details to register them in the system.")
    
    # Registration workflow
    if 'registration_step' not in st.session_state:
        st.session_state.registration_step = 1
    
    if st.session_state.registration_step == 1:
        capture_photo_from_upload()
    elif st.session_state.registration_step == 2:
        student_registration_form()
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔄 Start New Registration"):
            st.session_state.registration_step = 1
            st.session_state.captured_image = None
            st.session_state.student_data = {}
            if 'face_data' in st.session_state:
                del st.session_state.face_data
            st.rerun()
    
    with col2:
        if st.session_state.registration_step == 2:
            if st.button("⬅️ Back to Photo"):
                st.session_state.registration_step = 1
                st.rerun()

def show_register_attendance_page():
    """Show the Register Attendance page."""
    st.header("✅ Register Attendance")
    st.markdown("Mark student attendance by identifying faces from uploaded photos or live camera.")
    
    # Placeholder content
    st.info("🚧 This feature is coming soon!")
    st.markdown("""
    **Planned Features:**
    - 📷 Live camera attendance marking
    - 📋 Batch photo processing
    - ⏰ Real-time attendance tracking
    - 📊 Daily attendance reports
    """)

def show_attendance_analysis_page():
    """Show the Attendance Analysis page."""
    st.header("📊 Attendance Analysis")
    st.markdown("View detailed attendance statistics and generate reports.")
    
    # Placeholder content
    st.info("🚧 This feature is coming soon!")
    st.markdown("""
    **Planned Features:**
    - 📈 Attendance trends and statistics
    - 📅 Date range filtering
    - 👥 Student-wise attendance reports
    - 📊 Department-wise analytics
    - 📋 Export reports (PDF, Excel)
    """)

def main():
    """Main Streamlit application."""
    # Center-aligned page header
    st.markdown(
        '<h1 style="text-align: center; color: #1f77b4; margin-bottom: 2rem;">🎓 ArcFace Student Attendance System</h1>', 
        unsafe_allow_html=True
    )
    
    # Initialize system
    if not initialize_system():
        st.stop()
    
    # Sidebar navigation
    with st.sidebar:
        st.header("📊 System Status")
        if st.session_state.face_pipeline is not None:
            st.success("🟢 System Ready")
        else:
            st.warning("🟡 System Not Initialized")
        
        st.header("📈 Database Statistics")
        show_database_stats()
        
        st.markdown("---")
        st.header("🧭 Navigation")
        
        # Navigation buttons - all with same styling (no type="primary")
        if st.button("👤 Register Student", use_container_width=True):
            st.session_state.current_page = "register_student"
        
        if st.button("✅ Register Attendance", use_container_width=True):
            st.session_state.current_page = "register_attendance"
        
        if st.button("📊 Attendance Analysis", use_container_width=True):
            st.session_state.current_page = "attendance_analysis"
    
    # Initialize current page if not set
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "register_student"
    
    # Display selected page
    if st.session_state.current_page == "register_student":
        show_register_student_page()
    elif st.session_state.current_page == "register_attendance":
        show_register_attendance_page()
    elif st.session_state.current_page == "attendance_analysis":
        show_attendance_analysis_page()

if __name__ == "__main__":
    main()
