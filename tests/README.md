# Tests Directory

This directory contains all test files for the ArcFace face recognition system.

## Test Categories

### Core Model Tests
- `test_resnet.py` - ResNet backbone testing
- `test_enhanced_resnet.py` - Enhanced ResNet with device management and validation
- `test_cdcn.py` - CDCN anti-spoofing model testing
- `test_arcface_weight_loading.py` - ArcFace weight loading tests
- `test_enhanced_arcface_head.py` - Enhanced ArcFace head implementation tests
- `test_enhanced_arcface_recognizer.py` - Enhanced ArcFace recognizer tests

### Pipeline Tests
- `test_still_pipeline.py` - Still image pipeline testing
- `test_still_pipeline_visual.py` - Visual still image pipeline tests
- `test_still_pipeline_stub.py` - Pipeline stub testing
- `test_pipeline_integration.py` - Full pipeline integration tests
- `test_pipeline_validation.py` - Pipeline validation without weight files
- `test_live_video_visual.py` - Live video pipeline visual tests

### Utility Tests
- `test_alignment.py` - Face alignment utility tests
- `test_face_db.py` - Face database functionality tests
- `test_bounding_boxes.py` - Bounding box processing tests
- `test_auto_weight_mapping.py` - Automatic weight mapping tests

### Integration Tests
- `test_integration_enhanced.py` - Enhanced model integration tests
- `test_complete_system.py` - Complete system testing
- `test_instantiate.py` - Model instantiation tests
- `test_enhanced_features.py` - Enhanced feature testing
- `test_enhanced_retinaface.py` - Enhanced RetinaFace detector tests
- `test_fas_td.py` - Face anti-spoofing temporal detection tests

## Running Tests

### Run All Tests
```bash
# From the project root directory
python -m pytest tests/

# Or run from the tests directory
cd tests
python -m pytest .
```

### Run Specific Tests
```bash
# Run pipeline validation test
python tests/test_pipeline_validation.py

# Run enhanced model tests
python tests/test_enhanced_resnet.py
python tests/test_enhanced_arcface_recognizer.py
```

### Run Individual Test Categories
```bash
# Model tests
python -m pytest tests/test_*resnet*.py tests/test_*cdcn*.py tests/test_*arcface*.py

# Pipeline tests
python -m pytest tests/test_*pipeline*.py

# Utility tests
python -m pytest tests/test_alignment.py tests/test_face_db.py
```

## Test Requirements

- All tests require the conda environment `sars` to be activated
- Some tests require CUDA support for GPU testing
- Visual tests may require additional dependencies for image processing
- Integration tests require all model weights to be available

## Note

Tests are organized to be self-contained and can be run independently. The test files have been moved from the root directory to maintain a clean project structure.
