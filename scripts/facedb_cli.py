import argparse
import numpy as np
import torch
from pathlib import Path
import cv2

from models.retinaface import RetinaFaceDetector
from models.arcface import ArcFaceRecognizer
from utils.alignment import align_face
from utils.face_db import FaceDB


def load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


def extract_single_face(detector: RetinaFaceDetector, recognizer: ArcFaceRecognizer, img_path: str, device: str = 'cpu'):
    img = load_image(img_path)
    dets = detector.detect(img)
    if not dets:
        raise RuntimeError("No face detected")
    det = dets[0]
    x1, y1, x2, y2, score = det['box']
    face = img[y1:y2, x1:x2]
    if 'landmarks' in det:
        try:
            face = align_face(img, det['landmarks'])
        except Exception:
            pass
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (112, 112))
    tensor = torch.from_numpy(face_resized).permute(2,0,1).unsqueeze(0).float() / 255.0
    tensor = tensor.to(device)
    with torch.no_grad():
        emb = recognizer.extract(tensor).cpu().numpy()[0]
    return emb


def main():
    parser = argparse.ArgumentParser(description='FaceDB CLI for enrollment and search')
    parser.add_argument('--db', default='data/facedb', help='Database root directory')
    parser.add_argument('--weights', default=None, help='ArcFace weights path')
    parser.add_argument('--device', default='cpu')
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_enroll = sub.add_parser('enroll', help='Enroll an image')
    p_enroll.add_argument('--id', required=True, help='Person ID / label')
    p_enroll.add_argument('--image', required=True, help='Path to image file')
    p_enroll.add_argument('--update', action='store_true', help='Allow update averaging if ID exists')

    p_search = sub.add_parser('search', help='Search an image')
    p_search.add_argument('--image', required=True, help='Query image path')
    p_search.add_argument('--topk', type=int, default=5)

    p_match = sub.add_parser('match', help='Match an image with threshold')
    p_match.add_argument('--image', required=True, help='Query image path')
    p_match.add_argument('--threshold', type=float, default=0.35)

    args = parser.parse_args()

    device = args.device
    detector = RetinaFaceDetector(device=device)
    recognizer = ArcFaceRecognizer(embedding_size=512, weight_path=args.weights).to(device)
    db = FaceDB(args.db)

    if args.cmd == 'enroll':
        emb = extract_single_face(detector, recognizer, args.image, device=device)
        db.enroll(args.id, emb, allow_update=args.update)
        print(f"Enrolled {args.id}. Total entries: {db.count()}")
    elif args.cmd == 'search':
        emb = extract_single_face(detector, recognizer, args.image, device=device)
        results = db.search(emb, top_k=args.topk)
        for pid, score in results:
            print(f"{pid}\t{score:.4f}")
    elif args.cmd == 'match':
        emb = extract_single_face(detector, recognizer, args.image, device=device)
        pid, score = db.match(emb, threshold=args.threshold)
        print(f"match_id={pid} score={score:.4f}")

if __name__ == '__main__':
    main()
