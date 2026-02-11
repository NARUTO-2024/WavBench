from .base import VoiceAssistant
from .step_audio2 import StepAudio2Assistant

# Model class mapping for easy access
model_cls_mapping = {
    'step_audio2': StepAudio2Assistant,
}

__all__ = [
    'VoiceAssistant',
    'StepAudio2Assistant',
    'model_cls_mapping',
]
