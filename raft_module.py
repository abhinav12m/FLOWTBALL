import torch
import numpy as np
import cv2
import sys
import os

# Add RAFT path to import core files
RAFT_PATH = os.path.join(os.path.dirname(__file__), "RAFT")
sys.path.append(RAFT_PATH)

from core.raft import RAFT
from core.utils.utils import InputPadder
from torchvision import transforms
import argparse

class RAFTOpticalFlow:
    def __init__(self, model_path="RAFT/raft-things.pth", device="cpu"):
        self.device = device
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        import argparse

        args = argparse.Namespace(
            small=False,
            mixed_precision=False,
            alternate_corr=False
        )

        model = RAFT(args)

        # ✅ Load weights and remove "module." prefix
        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)

        model = model.to(self.device)
        model.eval()
        return model

    def preprocess(self, image1, image2):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image1 = transform(image1).unsqueeze(0).to(self.device)
        image2 = transform(image2).unsqueeze(0).to(self.device)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        return image1, image2

    def estimate_flow(self, frame1, frame2):
        """
        Estimate dense optical flow between two frames.
        Returns flow of shape (H, W, 2)
        """
        image1, image2 = self.preprocess(frame1, frame2)

        with torch.no_grad():
            _, flow_up = self.model(image1, image2, iters=20, test_mode=True)

        flow = flow_up[0].permute(1, 2, 0).cpu().numpy()
        return flow