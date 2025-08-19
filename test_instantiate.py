import torch
from pipelines.still_image_pipeline import StillImageFacePipeline
from pipelines.video_pipeline import LiveVideoFacePipeline


def main():
    still = StillImageFacePipeline()
    live = LiveVideoFacePipeline()
    print("Still pipeline device:", still.device)
    print("Live pipeline device:", live.device)
    # Create dummy face tensor to test recognizer & liveness forward pass
    dummy = torch.randn(1,3,112,112, device=still.device)
    emb = still.recognizer.extract(dummy)
    score, depth = still.liveness(dummy)
    print("Embedding shape:", emb.shape, "Liveness score tensor shape:", score.shape if hasattr(score,'shape') else type(score))

if __name__ == "__main__":
    main()
