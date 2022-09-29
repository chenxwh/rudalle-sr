import sys
import torch
import tempfile
import numpy as np
from PIL import Image
from cog import BasePredictor, Input, Path

sys.path.append("ru-dalle")
from rudalle.realesrgan.model import RealESRGAN


class Predictor(BasePredictor):
    def setup(self):
        device = torch.device("cuda:0")
        scales = [2, 4, 8]
        self.models = {}
        for scale in scales:
            model = RealESRGAN(device, scale)
            model.load_weights(f"models/RealESRGAN_x{scale}.pth")
            self.MODELS[scale] = model
        print("Model loaded!")

    def predict(
        self,
        image: Path = Input(description="Input image"),
        scale: int = Input(
            description="Choose up-scaling factor", default=4, choices=[2, 4, 8]
        ),
    ) -> Path:
        realesrgan = self.models[scale]
        input_image = Image.open(str(image))
        input_image = input_image.convert("RGB")
        with torch.no_grad():
            print("Up-scaling!")
            sr_image = realesrgan.predict(np.array(input_image))
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        sr_image.save(str(out_path))
        return out_path
