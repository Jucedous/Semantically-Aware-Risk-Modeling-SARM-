from __future__ import annotations
import matplotlib.pyplot as plt
from .ui_app import InteractiveLLMApp

def main(scene_path: str):
    app = InteractiveLLMApp(scene_path)
    plt.show(block=True)
