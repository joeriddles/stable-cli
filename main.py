from typing import Type
from enum import StrEnum
from pathlib import Path

import typer

from .models import Model, NitroModel, SpiderverseModel


class ModelChoice(StrEnum):
    spiderverse = "nitrosocke/spider-verse-diffusion"
    nitro = "nitrosocke/nitro-diffusion"

CHOICE_TO_MODEL: dict[ModelChoice, Type[Model]] = {
    ModelChoice.spiderverse: SpiderverseModel,
    ModelChoice.nitro: NitroModel,
}


def main(model: ModelChoice, prompt: str, filepath: str=""):
    print(model, prompt)

    model_ = CHOICE_TO_MODEL[model]()
    image = model_.run(prompt)

    filepath = filepath or str(Path(".", "output", prompt.replace(" ", "_")))
    image.save(f"{filepath}.png")


if __name__ == "__main__":
    typer.run(main)
