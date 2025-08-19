import numpy as np
import cv2
from utils.alignment import align_face

# Synthetic 5 landmarks approximate positions in a 112x112 face crop
# left_eye, right_eye, nose, left_mouth, right_mouth

def test_align_face_smoke():
    img = np.zeros((300,300,3), dtype=np.uint8)
    # fake landmarks (x,y)
    lms = [(120,130),(180,132),(150,160),(135,190),(170,188)]
    aligned = align_face(img, lms, output_size=112)
    assert aligned.shape == (112,112,3)
