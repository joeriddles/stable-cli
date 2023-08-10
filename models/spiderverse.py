from typing import Any

from diffusers import StableDiffusionPipeline
import torch

class SpiderverseModel:
    MODEL_ID = "nitrosocke/spider-verse-diffusion"

    def run(self, prompt: str) -> Any:
        pipe: Any = StableDiffusionPipeline.from_pretrained(self.MODEL_ID, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        if "spiderverse style" not in prompt:
            prompt += ", spiderverse style"

        image = pipe(prompt).images[0]
        return image

