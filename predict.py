import sys
import torch
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
import cog

sys.path.append("ru-dalle")
from rudalle.realesrgan.model import RealESRGAN


class Predictor(cog.Predictor):
    def setup(self):
        device = torch.device("cuda:0")
        scales = [2, 4, 8]
        self.models = {}
        for scale in scales:
            model = RealESRGAN(device, scale)
            model.load_weights(f"models/RealESRGAN_x{scale}.pth")
            self.MODELS[scale] = model
        print("Model loaded!")

    @cog.input(
        "image",
        type=Path,
        help="input image",
    )
    @cog.input(
        "scale",
        type=int,
        default=4,
        options=[2, 4, 8],
        help="choose up-scaling factor",
    )
    def predict(self, image, scale):
        realesrgan = self.models[scale]
        input_image = Image.open(str(image))
        with torch.no_grad():
            print("Up-scaling!")
            sr_image = realesrgan.predict(np.array(input_image))
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        sr_image.save(str(out_path))
        sr_image.save("000.png")
        return out_path
