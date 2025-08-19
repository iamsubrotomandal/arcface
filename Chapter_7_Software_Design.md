# Chapter 7: Software Design

## 7.1 Introduction
This chapter details the software architecture of the Face Recognition Attendance and Reporting System (SARS)—a streamlined, production-ready solution for secure, batch-processing student attendance using RetinaFace face detection and ArcFace face recognition. The system is designed for educational deployment, supporting efficient group photo processing, unknown face identification, and comprehensive attendance reporting.

SARS employs a simplified two-stage computer vision pipeline, integrating face detection (RetinaFace) and face recognition (ArcFace) optimized for batch processing of group photos. The architecture prioritizes efficiency, accuracy, and user-friendly operation, supporting future extensions and easy maintenance.

**Key Features:**
- Streamlit-based multi-page UI for registration, batch attendance processing, and reporting
- RetinaFace-powered face detection optimized for group photos
- ArcFace-based high-precision face recognition with ONNX optimization
- Comprehensive unknown face identification with visual feedback
- SQLite3 for persistent, indexed attendance/session storage
- CSV export and ZIP download capabilities for results
- Real-time progress tracking and batch processing optimization
- Robust error handling and user feedback

## 7.2 System Architecture Overview
SARS is designed for batch processing with sub-second per-face latency, maintaining high throughput for group photo scenarios. The system comprises the following streamlined functional modules:

- **Streamlit UI Module:** Clean web interface for registration, batch upload, and results visualization
- **RetinaFace Detection Module:** Robust face detection optimized for group photos and challenging conditions
- **ArcFace Recognition Module:** High-precision face embedding extraction and similarity matching
- **Batch Processing Module:** Efficient handling of multiple images, ZIP files, and concurrent face processing
- **Database Module:** SQLite3 for storing students, embeddings, sessions, and attendance records
- **Results Visualization Module:** Comprehensive display of recognized students, unknown faces, and attendance logs

**Performance Optimizations:**
- Vectorized similarity calculations for batch face matching
- ONNX-optimized ArcFace models for production deployment
- Efficient memory management for large batch processing
- Progress tracking and user feedback for long-running operations

**Error Handling:**
- Graceful handling of detection and recognition failures
- User-friendly error messages and recovery suggestions
- Fallback mechanisms for model loading and processing errors

*Figure 7.1: Simplified System Architecture*

## 7.3 Module Design

### 7.3.1 User Interface Module (Streamlit)
The Streamlit-based UI provides a clean, intuitive interface for all system operations:

**Registration Page:**
- Student information form with validation
- Image upload for face registration
- Real-time feedback and success confirmation

**Batch Attendance Page:**
- Session metadata input (date, program, lecture, session, lecturer)
- Multi-file upload with ZIP support
- Similarity threshold configuration
- Real-time processing progress
- Comprehensive results visualization

**Results Display:**
- Recognized students table with similarity scores
- Unknown faces grid with cropped images
- Attendance log viewer with session filtering
- CSV and ZIP export functionality

### 7.3.2 Face Detection Module (RetinaFace)
RetinaFace provides robust, single-stage face detection optimized for group photos:

**Core Functions:**
```python
detect_faces(image) -> Dict[str, Any]
extract_facial_areas(faces_dict) -> List[Tuple[int, int, int, int]]
crop_faces(image, facial_areas) -> List[np.ndarray]
```

**Key Features:**
- Multi-face detection in group photos
- Robust performance in challenging lighting conditions
- Automatic face area extraction and validation
- Efficient batch processing of multiple images

### 7.3.3 Face Recognition Module (ArcFace)
ArcFace provides high-precision face recognition with angular margin optimization:

**Core Functions:**
```python
extract_embedding(face_image) -> np.ndarray
calculate_similarity_batch(embeddings, reference_matrix) -> np.ndarray
match_faces(embeddings, database, threshold) -> List[Match]
```

**Key Features:**
- ONNX-optimized models for production deployment
- PyTorch fallback for development environments
- Vectorized similarity calculations for batch processing
- Configurable similarity thresholds

### 7.3.4 Database Module (SQLite)
Normalized database schema supporting efficient attendance management:

**Core Tables:**
- `students`: Student registration information
- `student_information`: Extended student details (SRN, program, etc.)
- `embeddings`: Face embeddings for recognition
- `attendance`: Session-based attendance records

**Key Functions:**
```python
load_embeddings() -> List[Tuple[int, str, np.ndarray]]
log_attendance_session_once() -> bool
attendance_for_session() -> List[AttendanceRecord]
```

## 7.4 Data Flow Architecture

### 7.4.1 Registration Flow
1. Student submits registration form with personal information
2. Student uploads face images through Streamlit interface
3. RetinaFace detects and validates faces in uploaded images
4. ArcFace extracts embeddings from detected faces
5. Student record and embeddings stored in database
6. Success confirmation displayed to user

### 7.4.2 Batch Attendance Flow
1. User uploads images (single files or ZIP archives)
2. System unpacks and validates uploaded images
3. RetinaFace detects all faces in each image
4. ArcFace extracts embeddings for detected faces
5. Embeddings matched against registered student database
6. Recognized students logged for session attendance
7. Unknown faces identified and displayed for review
8. Comprehensive results presented with export options

### 7.4.3 Results Visualization Flow
1. Recognized students displayed in sortable table
2. Unknown faces shown in grid layout with similarity scores
3. Attendance logs accessible through expandable interface
4. CSV export for recognized students data
5. ZIP download for unknown face images

## 7.5 Performance Considerations

### 7.5.1 Scalability
- Vectorized similarity calculations reduce computation time
- Batch processing optimization for multiple face handling
- Efficient memory management for large image uploads
- Progress tracking prevents UI blocking during processing

### 7.5.2 Accuracy Optimization
- RetinaFace ensures robust multi-face detection
- ArcFace provides high-precision recognition with angular margin
- Configurable similarity thresholds for different scenarios
- Unknown face identification prevents false positives

### 7.5.3 User Experience
- Real-time progress indicators for batch processing
- Clear error messages and recovery guidance
- Intuitive interface design with minimal learning curve
- Comprehensive results visualization with export capabilities

## 7.6 Security and Privacy
- Local processing with no external API dependencies
- SQLite database with local storage only
- No face images stored permanently (only embeddings)
- Session-based data management with cleanup procedures

## 7.7 Deployment Architecture
The system is designed for easy deployment in educational environments:
- Single Python application with Streamlit frontend
- Minimal dependencies with conda/pip installation
- Local SQLite database for zero-configuration setup
- Cross-platform compatibility (Windows, macOS, Linux)

## 7.8 Future Extensions
The modular architecture supports future enhancements:
- Integration with Learning Management Systems (LMS)
- Advanced analytics and engagement metrics
- Multi-camera input support
- Enhanced reporting and visualization features
- API endpoints for external system integration
