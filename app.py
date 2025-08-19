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
    from utils.database import StudentDatabase
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

# Hide Streamlit's default form submission text using config
import streamlit.components.v1 as components

# Inject CSS and JavaScript directly into the page head
components.html("""
<style>
/* ABSOLUTE NUCLEAR OPTION - Hide ALL text in forms */
.stForm .stMarkdown, 
.stForm p, 
.stForm span:not(button span),
.stForm div[data-testid="stMarkdownContainer"],
.stForm div[data-testid="stText"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    height: 0 !important;
    width: 0 !important;
    overflow: hidden !important;
}

/* Force hide any element containing the text */
.stForm *:contains("Press"), 
.stForm *:contains("Enter"),
.stForm *:contains("submit") {
    display: none !important;
}

/* Style buttons properly */
.stForm button[kind="primaryFormSubmit"] {
    background-color: #1f77b4 !important;
    border-color: #1f77b4 !important;
    color: white !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    border-radius: 6px !important;
}

.stForm button[kind="primaryFormSubmit"]:hover {
    background-color: #1565c0 !important;
    border-color: #1565c0 !important;
}

.stForm button span {
    color: white !important;
    font-weight: 600 !important;
}
</style>

<script>
// Most aggressive approach - override Streamlit's form rendering
(function() {
    // Function to completely remove placeholder text
    function nukeFormText() {
        // Remove ALL text nodes in forms
        const forms = document.querySelectorAll('.stForm');
        forms.forEach(form => {
            const walker = document.createTreeWalker(
                form,
                NodeFilter.SHOW_TEXT,
                null,
                false
            );
            
            const textNodes = [];
            let node;
            while (node = walker.nextNode()) {
                textNodes.push(node);
            }
            
            textNodes.forEach(textNode => {
                if (textNode.textContent.includes('Press') || 
                    textNode.textContent.includes('Enter') ||
                    textNode.textContent.includes('submit')) {
                    textNode.textContent = '';
                    textNode.parentNode.remove();
                }
            });
        });
        
        // Remove specific elements
        const elementsToRemove = document.querySelectorAll(
            '.stForm p, .stForm .stMarkdown, .stForm div[data-testid="stMarkdownContainer"]'
        );
        elementsToRemove.forEach(el => {
            if (el.textContent.includes('Press') || 
                el.textContent.includes('Enter') ||
                el.textContent.includes('submit')) {
                el.remove();
            }
        });
    }
    
    // Disable Enter key completely
    function disableEnter() {
        document.addEventListener('keydown', function(e) {
            if ((e.key === 'Enter' || e.keyCode === 13) && 
                e.target.closest('.stForm')) {
                e.preventDefault();
                e.stopPropagation();
                e.stopImmediatePropagation();
                return false;
            }
        }, true);
    }
    
    // Run immediately
    nukeFormText();
    disableEnter();
    
    // Run on every possible event
    ['DOMContentLoaded', 'load', 'resize', 'scroll'].forEach(event => {
        window.addEventListener(event, () => {
            setTimeout(nukeFormText, 0);
            setTimeout(nukeFormText, 100);
            setTimeout(nukeFormText, 500);
        });
    });
    
    // Ultra-aggressive mutation observer
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            nukeFormText();
        });
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        characterData: true,
        attributeOldValue: true,
        characterDataOldValue: true
    });
    
    // Continuous removal
    setInterval(nukeFormText, 50);
})();
</script>
""", height=0)

# Additional Streamlit config to suppress form text
try:
    import streamlit.config as config
    config.set_option('theme.base', 'light')
except:
    pass

# Initialize session state
if 'face_pipeline' not in st.session_state:
    st.session_state.face_pipeline = None
if 'face_db' not in st.session_state:
    st.session_state.face_db = None
if 'student_db' not in st.session_state:
    st.session_state.student_db = None
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
            
            # Initialize student database
            st.session_state.student_db = StudentDatabase()
            
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
    if st.session_state.student_db is not None:
        try:
            # Get database statistics
            stats = st.session_state.student_db.get_database_stats()
            
            # Display main metrics
            st.metric("Total Students", stats['total_students'])
            st.metric("Recent Registrations (7 days)", stats['recent_registrations'])
            
            # Show program distribution
            if stats['programs']:
                with st.expander("Students by Program"):
                    for program, count in stats['programs'].items():
                        st.text(f"• {program}: {count}")
            
            # Show recent students
            if stats['total_students'] > 0:
                with st.expander("Recent Students"):
                    recent_students = st.session_state.student_db.get_all_students()[:5]
                    for student in recent_students:
                        st.text(f"• {student['student_id']} - {student['first_name']} {student['last_name']}")
                        
        except Exception as e:
            st.error(f"Error reading database: {str(e)}")
    else:
        st.metric("Total Students", 0)

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
    """Display student registration confirmation and final details."""
    st.subheader("🎓 Confirm Registration Details")
    
    # Show face analysis results
    if 'face_data' in st.session_state:
        face_data = st.session_state.face_data
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Liveness Score", f"{face_data['liveness_score']:.3f}")
        with col2:
            st.metric("Face Quality", "✅ Good" if face_data['liveness_score'] > 0.7 else "⚠️ Fair")
    
    # Display pre-filled student information
    if 'student_data' in st.session_state:
        student_info = st.session_state.student_data
        
        st.markdown("### 📋 Student Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.text_input("First Name", value=student_info.get('first_name', ''), disabled=True)
            st.text_input("Date of Birth", value=student_info.get('date_of_birth', ''), disabled=True)
            st.text_input("Phone", value=student_info.get('phone', ''), disabled=True)
        with col2:
            st.text_input("Middle Name", value=student_info.get('middle_name', ''), disabled=True)
            st.text_input("Gender", value=student_info.get('gender', ''), disabled=True)
            st.text_input("Email", value=student_info.get('email', ''), disabled=True)
        with col3:
            st.text_input("Last Name", value=student_info.get('last_name', ''), disabled=True)
            st.text_input("Student ID (SRN)", value=student_info.get('student_id', ''), disabled=True)
            st.text_input("Program", value=student_info.get('program', ''), disabled=True)
    
    # Additional fields form
    with st.form("final_registration"):
        st.markdown("### 📝 Additional Information")
        
        col1, col2 = st.columns(2)
        with col1:
            year = st.selectbox("Academic Year *", ["", "1st Year", "2nd Year", "3rd Year", "4th Year", "Graduate"])
            emergency_contact = st.text_input("Emergency Contact")
        with col2:
            semester = st.selectbox("Current Semester *", ["", "1", "2", "3", "4", "5", "6", "7", "8"])
            guardian_name = st.text_input("Guardian Name")
        
        address = st.text_area("Address")
        notes = st.text_area("Additional Notes")
        
        # Submit button - centered
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit_button = st.form_submit_button(
                "🎓 Complete Registration", 
                type="primary",
                use_container_width=True
            )
        
        if submit_button:
            # Validate required fields
            if not all([year and year.strip(), semester and semester.strip()]):
                st.error("❌ Please fill in all required fields marked with *")
            else:
                # Combine all student data
                final_student_data = st.session_state.student_data.copy()
                final_student_data.update({
                    'year': year,
                    'semester': semester,
                    'emergency_contact': emergency_contact,
                    'guardian_name': guardian_name,
                    'address': address,
                    'notes': notes,
                    'registration_date': time.strftime('%Y-%m-%d %H:%M:%S')
                })
                
                # Register student
                if register_student(final_student_data):
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
    """Register student in both face database and student database."""
    try:
        if 'face_data' not in st.session_state:
            st.error("No face data available")
            return False
        
        # Create identity string for face database
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
        
        # Save to student database
        if st.session_state.student_db:
            db_success = st.session_state.student_db.insert_student(student_data)
            if not db_success:
                st.error("Failed to save student information to database")
                return False
        
        return True
            
    except Exception as e:
        st.error(f"Registration failed: {str(e)}")
        return False

def show_register_student_page():
    """Show the Register New Student page."""
    st.markdown(
        '<h2 style="text-align: center; color: #1f77b4; margin-bottom: 1rem;">👤 Register New Student</h2>', 
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; color: #666; margin-bottom: 2rem;">Upload a student photo and fill in their details to register them in the system.</p>',
        unsafe_allow_html=True
    )
    
    # Student Information Form
    st.subheader("📝 Student Information")
    
    with st.form("student_info_form"):
        # First row: Names
        col1, col2, col3 = st.columns(3)
        with col1:
            first_name = st.text_input("First Name *")
        with col2:
            middle_name = st.text_input("Middle Name")
        with col3:
            last_name = st.text_input("Last Name *")
        
        # Second row: Personal details
        col1, col2, col3 = st.columns(3)
        with col1:
            dob = st.date_input("Date of Birth *")
        with col2:
            gender = st.selectbox("Gender *", ["", "Male", "Female", "Other", "Prefer not to say"])
        with col3:
            student_id = st.text_input("Student ID (SRN) *")
        
        # Third row: Contact details
        col1, col2 = st.columns(2)
        with col1:
            phone = st.text_input("Phone *")
        with col2:
            email = st.text_input("Email *")
        
        # Fourth row: Program
        program = st.selectbox("Program *", [
            "",
            "Artificial Intelligence",
            "Business Analytics", 
            "Cyber Security"
        ])
        
        # Form submission - centered button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            form_submitted = st.form_submit_button(
                "📝 Save Information & Proceed to Photo Upload", 
                use_container_width=True,
                type="primary"
            )
        
        if form_submitted:
            # Validate required fields
            if not all([first_name, last_name, dob, gender and gender.strip(), student_id, phone, email, program and program.strip()]):
                st.error("❌ Please fill in all required fields marked with *")
            elif st.session_state.student_db:
                # Check for duplicate Student ID
                existing_student = st.session_state.student_db.get_student(student_id)
                if existing_student:
                    st.error(f"❌ Student ID {student_id} already exists!")
                # Check for duplicate email
                elif st.session_state.student_db.check_email_exists(email):
                    st.error(f"❌ Email {email} is already registered!")
                else:
                    # Store student data in session state
                    st.session_state.student_data = {
                        'first_name': first_name,
                        'middle_name': middle_name,
                        'last_name': last_name,
                        'date_of_birth': str(dob),
                        'gender': gender,
                        'student_id': student_id,
                        'phone': phone,
                        'email': email,
                        'program': program
                    }
                    st.session_state.registration_step = 1
                    st.success("✅ Information saved! Please proceed to photo upload.")
                    st.rerun()
            else:
                st.error("❌ Database not initialized. Please refresh the page.")
    
    # Show photo upload section only if student info is saved
    if 'student_data' in st.session_state and st.session_state.student_data:
        st.markdown("---")
        
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
    st.markdown(
        '<h2 style="text-align: center; color: #1f77b4; margin-bottom: 1rem;">✅ Register Attendance</h2>', 
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; color: #666; margin-bottom: 2rem;">Mark student attendance by identifying faces from uploaded photos or live camera.</p>',
        unsafe_allow_html=True
    )
    
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
    st.markdown(
        '<h2 style="text-align: center; color: #1f77b4; margin-bottom: 1rem;">📊 Attendance Analysis</h2>', 
        unsafe_allow_html=True
    )
    st.markdown(
        '<p style="text-align: center; color: #666; margin-bottom: 2rem;">View detailed attendance statistics and generate reports.</p>',
        unsafe_allow_html=True
    )
    
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
    
    # Simple CSS to make form buttons blue
    st.markdown("""
        <style>
        .stForm button[kind="primaryFormSubmit"] {
            background-color: #1f77b4 !important;
            border-color: #1f77b4 !important;
            color: white !important;
        }
        .stForm button[kind="primaryFormSubmit"]:hover {
            background-color: #1565c0 !important;
            border-color: #1565c0 !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
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
