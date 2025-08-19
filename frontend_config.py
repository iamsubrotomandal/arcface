"""
Frontend Configuration for Face Recognition System
"""

# Streamlit configuration
STREAMLIT_CONFIG = {
    'page_title': 'Face Recognition System',
    'page_icon': '🎓',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Face recognition settings
FACE_RECOGNITION_CONFIG = {
    'match_threshold': 0.35,
    'liveness_threshold': 0.7,
    'min_face_size': 20,
    'confidence_threshold': 0.6
}

# Database settings
DATABASE_CONFIG = {
    'face_db_path': 'face_database.pkl',
    'backup_enabled': True,
    'backup_interval': 3600  # seconds
}

# UI settings
UI_CONFIG = {
    'max_image_size': 5 * 1024 * 1024,  # 5MB
    'supported_formats': ['jpg', 'jpeg', 'png'],
    'preview_size': (400, 400),
    'thumbnail_size': (150, 150)
}

# Department options
DEPARTMENTS = [
    "Computer Science",
    "Engineering", 
    "Mathematics",
    "Physics",
    "Chemistry",
    "Biology",
    "Business",
    "Psychology",
    "Art & Design",
    "Medicine",
    "Law",
    "Education",
    "Other"
]

# Year options
ACADEMIC_YEARS = [
    "1st Year",
    "2nd Year", 
    "3rd Year",
    "4th Year",
    "Graduate",
    "PhD"
]

# Theme colors
THEME_COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'success': '#28A745',
    'warning': '#FFC107',
    'error': '#DC3545',
    'info': '#007BFF'
}
