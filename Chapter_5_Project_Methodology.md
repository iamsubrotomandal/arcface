# Chapter 5: Project Methodology

## 5.1 Introduction

This project adopts a streamlined methodology inspired by the CRISP-DM framework, customized for the development of a robust, batch-processing student attendance system using RetinaFace face detection and ArcFace face recognition. The approach integrates state-of-the-art computer vision technologies with user-centric interface design to ensure accurate, efficient, and user-friendly attendance logging from group photos and batch uploads. The methodology emphasizes simplicity, performance, and practical deployment in educational settings.

## 5.2 Problem Understanding

The initial phase focused on understanding the core challenges of automating attendance in academic environments—specifically, the need for reliable face detection in group photos, high-precision face recognition, and efficient batch processing of uploaded images. Stakeholders such as faculty, students, and administrators were considered to ensure the system addresses real-world requirements, including ease of use, accuracy, and the ability to identify unknown faces for potential enrollment.

## 5.3 Data Collection and Preprocessing

A dataset of student face images is collected through the system's registration module, which supports image uploads through the Streamlit interface. Each student provides images under varying conditions to improve recognition robustness. Preprocessing steps include RetinaFace-based face detection, automatic face cropping, resizing to 160x160 pixels, and normalization. These steps standardize input for ArcFace embedding extraction and ensure consistent recognition performance.

## 5.4 Face Detection and Embedding Extraction

The system employs RetinaFace for robust face detection, particularly excelling at detecting multiple faces in group photos and challenging lighting conditions. For embedding extraction, the system uses ArcFace with ONNX optimization for production deployment, with a PyTorch fallback for development environments. This approach ensures high-precision face recognition while maintaining computational efficiency. Embeddings are stored in a SQLite database with vectorized similarity calculations for fast batch matching.

## 5.5 Batch Processing and Unknown Face Detection

The core workflow focuses on batch processing of uploaded images (including ZIP files) containing single or multiple faces. RetinaFace detects all faces in each image, crops them individually, and passes them through the ArcFace recognition pipeline. Faces that don't match registered students above the similarity threshold are classified as "unknown" and presented with cropped images for potential enrollment identification. This workflow is optimized for educational scenarios where group photos are common.

## 5.6 Attendance Logging and Session Management

Attendance is logged through the simplified batch upload flow, where each recognized student is logged once per session. Each attendance event is associated with session metadata (date, program, session, lecture, lecturer) and student information (name, SRN). The system enforces per-session deduplication, ensuring each student is logged only once per session. All data is persisted in a normalized SQLite schema with efficient indexing for fast retrieval.

## 5.7 Streamlit-Based User Interface

A clean, multi-page Streamlit UI orchestrates student registration, batch attendance processing, and results visualization. The interface includes real-time feedback, required field validation, progress indicators, and comprehensive results display. Key features include similarity threshold adjustment, recognized students table with CSV export, unknown faces grid with ZIP download, and session attendance logs with expandable views.

## 5.8 Results Visualization and Export

The system provides comprehensive results visualization including:
- **Recognized Students:** Tabular display with similarity scores and CSV export functionality
- **Unknown Faces:** Grid layout with similarity scores and ZIP download of cropped face images
- **Attendance Logs:** Session-specific attendance records with timestamp and similarity data
- **Processing Summary:** Total faces detected, recognized, and unknown counts with success metrics

## 5.9 Challenges and Resolution

Key challenges encountered included:
- **Group photo face detection:** Addressed by implementing RetinaFace for superior multi-face detection performance
- **Computational efficiency:** Mitigated with ONNX-optimized ArcFace models and vectorized similarity calculations
- **Unknown face handling:** Resolved by implementing comprehensive unknown face identification with visual feedback
- **Batch processing scalability:** Ensured through efficient image handling, progress tracking, and memory management
- **User experience:** Enhanced with intuitive interface design, clear progress indicators, and comprehensive result presentation

## 5.10 Conclusion

This methodology integrates state-of-the-art RetinaFace face detection and ArcFace recognition into a practical, user-friendly attendance system optimized for batch processing and group photos. The streamlined approach ensures the solution is both technically robust and operationally effective for educational environments. The system successfully addresses the core requirements of accurate face detection, high-precision recognition, and comprehensive unknown face identification, making it suitable for modern educational attendance management scenarios.
