from typing import Any

from diffusers import StableDiffusionPipeline
import torch

class NitroModel:
    MODEL_ID = "nitrosocke/nitro-diffusion"

    def run(self, prompt: str) -> Any:
        pipe: Any = StableDiffusionPipeline.from_pretrained(self.MODEL_ID, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        VALID_STYLES = set(["archer style", "arcane style", "modern disney style"])
        if all([style not in prompt for style in VALID_STYLES]):
            raise SystemExit(f"You must include one or more of the following styles: {', '.join(VALID_STYLES)}")

        image = pipe(prompt).images[0]
        return image
