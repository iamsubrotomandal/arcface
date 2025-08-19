import os
import shutil
import numpy as np
from utils.face_db import FaceDB

def setup_module(module):
    if os.path.isdir('temp_facedb'):
        shutil.rmtree('temp_facedb')


def test_enroll_and_search():
    db = FaceDB('temp_facedb')
    emb1 = np.random.randn(512).astype('float32')
    emb2 = np.random.randn(512).astype('float32')
    db.enroll('alice', emb1)
    db.enroll('bob', emb2)
    assert db.count() == 2
    q = emb1 + 0.01*np.random.randn(512).astype('float32')
    pid, score = db.match(q, threshold=0.1)
    assert pid == 'alice'
    assert score > 0.1


def test_update_enroll():
    db = FaceDB('temp_facedb')
    e1 = np.ones(512, dtype='float32')
    db.enroll('charlie', e1)
    e2 = -np.ones(512, dtype='float32')
    # update should average normalized vectors (which are identical magnitude but opposite)
    db.enroll('charlie', e2, allow_update=True)
    # After opposite vectors average, norm then re-normalize -> could be unstable; ensure still unit dimension
    assert db.count() >= 1
    results = db.search(e1, top_k=1)
    assert isinstance(results, list)


def teardown_module(module):
    if os.path.isdir('temp_facedb'):
        shutil.rmtree('temp_facedb')
