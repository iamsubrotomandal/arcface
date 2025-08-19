# Chapter 6: Resource Requirement Specification

## 6.1 Introduction
This chapter outlines the software and hardware resources required for the development, deployment, and operation of the Face Recognition Attendance and Reporting System. The system integrates RetinaFace face detection, ArcFace face recognition, and a Streamlit-based user interface, all optimized for batch processing and group photo scenarios in educational environments.

## 6.2 Software Requirements

### 6.2.1 User Interface
The user interface is developed using the Streamlit framework, enabling rapid creation of interactive, web-based applications. Streamlit provides real-time feedback, multi-page navigation, progress tracking, and supports batch image upload with comprehensive results visualization.

### 6.2.2 Face Detection and Recognition
The following Python libraries and frameworks are used for face detection and recognition:
- **RetinaFace**: State-of-the-art single-stage face detector optimized for group photos and challenging conditions
- **ArcFace (ONNX)**: High-precision face recognition with angular margin optimization for production deployment
- **PyTorch**: Fallback framework for ArcFace models in development environments
- **OpenCV**: For image preprocessing, color conversion, resizing, and basic computer vision operations
- **NumPy**: For efficient numerical operations, vectorized calculations, and array manipulations
- **ONNX Runtime**: For optimized model inference in production environments

### 6.2.3 Batch Processing and File Handling
- **Zipfile**: For handling ZIP archives containing multiple student photos
- **Hashlib**: For content-based deduplication of uploaded images
- **IO Libraries**: For efficient memory management during batch processing
- **Progress Tracking**: Real-time feedback during long-running batch operations

### 6.2.4 Database and Data Management
- **SQLite3**: Lightweight, file-based relational database for storing student, embedding, and attendance records
- **Pandas**: For data manipulation, attendance reporting, and CSV export functionality
- **Database Indexing**: Optimized queries for fast embedding retrieval and similarity matching

### 6.2.5 Results Visualization and Export
- **Pandas**: Used for generating recognized students tables, attendance summaries, and CSV exports
- **Streamlit Components**: For displaying unknown faces grid, similarity scores, and download buttons
- **Zipfile**: For creating downloadable archives of unknown face images
- **Image Processing**: For thumbnail generation and face crop visualization

### 6.2.6 Logging and Error Handling
- **Python logging module**: Captures runtime events, processing statistics, and error information
- **Exception Handling**: Robust error recovery with user-friendly feedback messages
- **Validation Logic**: Input validation for uploaded files, session metadata, and system parameters

#### Table 6.1: Python Package Versions (Updated)
| Sl. No. | Package           | Version    | Purpose                           |
|---------|-------------------|------------|-----------------------------------|
| 1       | streamlit         | 1.36.0     | Web UI framework                  |
| 2       | opencv-python     | 4.10.0.84  | Image processing                  |
| 3       | retina-face       | 0.0.15     | Face detection                    |
| 4       | facenet-pytorch   | 2.6.0      | ArcFace embeddings (fallback)     |
| 5       | onnxruntime       | Latest     | ONNX model inference              |
| 6       | torch             | Latest     | PyTorch backend                   |
| 7       | numpy             | 1.26.4     | Numerical operations              |
| 8       | pandas            | 2.2.2      | Data manipulation                 |
| 9       | scikit-learn      | 1.5.1      | Similarity calculations           |
| 10      | sqlite3           | stdlib     | Database management               |
| 11      | zipfile           | stdlib     | Archive handling                  |
| 12      | hashlib           | stdlib     | Content deduplication             |

## 6.3 Hardware Requirements

### 6.3.1 Minimum System Requirements
- **Processor**: Intel Core i5 or AMD Ryzen 5 (4 cores minimum)
- **RAM**: 8 GB (for batch processing of moderate-sized groups)
- **Storage**: 10 GB free space (for models, database, and temporary files)
- **Network**: Internet connection for initial model downloads
- **Display**: 1920x1080 resolution for optimal UI experience

### 6.3.2 Recommended System Requirements
- **Processor**: Intel Core i7 or AMD Ryzen 7 (8+ cores for large batch processing)
- **RAM**: 16 GB or higher (for processing large group photos and ZIP archives)
- **Storage**: SSD with 20+ GB free space (for improved model loading and processing speed)
- **GPU**: Optional CUDA-compatible GPU for accelerated processing (NVIDIA GTX 1060 or better)
- **Network**: Stable internet for model downloads and updates

### 6.3.3 Production Deployment Requirements
- **Server**: Multi-core CPU with 16+ GB RAM for concurrent user support
- **Storage**: SSD storage for database and model files
- **Backup**: Regular database backup capabilities
- **Monitoring**: System monitoring for performance and error tracking

## 6.4 Development Environment

### 6.4.1 Integrated Development Environment (IDE)
- **Visual Studio Code**: Primary IDE with Python extensions
- **Jupyter Notebooks**: For experimentation and model testing
- **Git**: Version control for code management
- **Conda/Pip**: Package management for dependency handling

### 6.4.2 Model Management
- **Model Storage**: Local directory structure for ONNX and PyTorch models
- **Automatic Downloads**: Fallback mechanisms for missing models
- **Version Control**: Model versioning for reproducible deployments
- **Optimization**: Model quantization and optimization for production

## 6.5 Deployment Architecture

### 6.5.1 Local Deployment
- **Single Machine**: Complete system running on local hardware
- **Database**: SQLite file-based storage
- **Models**: Local model files with automatic fallbacks
- **UI**: Streamlit server accessible via web browser

### 6.5.2 Educational Institution Deployment
- **Network Access**: Local network deployment for multiple users
- **Shared Database**: Centralized SQLite database for attendance records
- **Load Balancing**: Multiple Streamlit instances for concurrent access
- **Backup Strategy**: Regular database backups and model synchronization

## 6.6 Security and Privacy Requirements

### 6.6.1 Data Security
- **Local Processing**: All face processing performed locally
- **No External APIs**: Complete offline operation capability
- **Database Encryption**: Optional SQLite encryption for sensitive data
- **Access Control**: Session-based access management

### 6.6.2 Privacy Protection
- **Embedding Storage**: Only face embeddings stored, not original images
- **Data Retention**: Configurable data retention policies
- **Anonymization**: Option to anonymize attendance records
- **Audit Trails**: Comprehensive logging for compliance requirements

## 6.7 Performance Considerations

### 6.7.1 Batch Processing Optimization
- **Memory Management**: Efficient handling of large image batches
- **Progress Tracking**: Real-time feedback for long-running operations
- **Error Recovery**: Graceful handling of processing failures
- **Resource Monitoring**: CPU and memory usage optimization

### 6.7.2 Scalability Requirements
- **Concurrent Users**: Support for multiple simultaneous sessions
- **Database Performance**: Optimized queries for large datasets
- **Model Loading**: Efficient model initialization and caching
- **Result Export**: Fast CSV and ZIP generation for large result sets

## 6.8 Conclusion
The resource requirements for the Face Recognition Attendance and Reporting System are optimized for educational deployment with emphasis on batch processing capabilities. The combination of RetinaFace and ArcFace provides excellent accuracy while maintaining reasonable computational requirements. The system is designed to scale from single-user deployments to institution-wide implementations with appropriate hardware scaling.
