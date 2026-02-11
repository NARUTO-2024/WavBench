#!/usr/bin/env python3
"""
Step-Audio2 Voice Assistant Wrapper

Provides a unified interface for Step-Audio2 model inference,
supporting single-round, multi-round conversations, and audio output.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .base import VoiceAssistant

# Add src_step_audio2 to path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_step_audio2_path = os.path.join(_current_dir, 'src_step_audio2')
if _step_audio2_path not in sys.path:
    sys.path.insert(0, _step_audio2_path)

from stepaudio2 import StepAudio2
from token2wav import Token2wav


class StepAudio2Assistant(VoiceAssistant):
    """
    Step-Audio2 voice assistant wrapper for WavBench evaluation.

    Supports:
    - Single-round inference (text-only or with audio output)
    - Multi-round conversation (4 rounds with history)
    - Audio file input
    """

    # Default prompt for voice assistant
    DEFAULT_PROMPT = "You are a helpful assistant."
    AUDIO_ANALYSIS_PROMPT = "You are an expert in audio analysis. Please analyze the audio content and answer accurately."

    def __init__(
        self,
        model_path: str = None,
        token2wav_path: str = None,
        system_prompt: str = None,
    ):
        """
        Initialize Step-Audio2 assistant.

        Args:
            model_path: Path to Step-Audio2 model weights
            token2wav_path: Path to Token2wav model weights
            system_prompt: System prompt for the model
        """
        # Default paths from environment or hardcoded
        if model_path is None:
            model_path = os.environ.get(
                'STEP_AUDIO2_MODEL_PATH',
                '/data2/chenyifu/lyz/Step-Audio-2-mini'
            )
        if token2wav_path is None:
            token2wav_path = os.environ.get(
                'STEP_AUDIO2_TOKEN2WAV_PATH',
                os.path.join(model_path, 'token2wav')
            )

        self.model_path = model_path
        self.token2wav_path = token2wav_path
        self.system_prompt = system_prompt or self.DEFAULT_PROMPT

        # Lazy loading flags
        self._model = None
        self._token2wav = None

        # Default generation parameters (consistent with reference implementation)
        self.audio_temperature = 0.7
        self.audio_top_p = 0.9
        self.text_temperature = 0.7
        self.text_top_p = 0.9
        self.max_tokens = 2048

        # Default prompt wav for TTS
        self._prompt_wav = os.path.join(_step_audio2_path, 'assets/default_female.wav')

    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            print(f"Loading Step-Audio2 model from: {self.model_path}")
            self._model = StepAudio2(self.model_path)
            print("Step-Audio2 model loaded successfully")
        return self._model

    def _load_token2wav(self):
        """Lazy load token2wav."""
        if self._token2wav is None:
            print(f"Loading Token2wav from: {self.token2wav_path}")
            self._token2wav = Token2wav(self.token2wav_path)
            print("Token2wav loaded successfully")
        return self._token2wav

    @property
    def model(self):
        return self._load_model()

    @property
    def token2wav(self):
        return self._load_token2wav()

    def set_system_prompt(self, prompt: str):
        """Set the system prompt."""
        self.system_prompt = prompt

    def set_generation_params(
        self,
        audio_temperature: float = None,
        audio_top_p: float = None,
        text_temperature: float = None,
        text_top_p: float = None,
        max_tokens: int = None,
    ):
        """Set generation parameters."""
        if audio_temperature is not None:
            self.audio_temperature = audio_temperature
        if audio_top_p is not None:
            self.audio_top_p = audio_top_p
        if text_temperature is not None:
            self.text_temperature = text_temperature
        if text_top_p is not None:
            self.text_top_p = text_top_p
        if max_tokens is not None:
            self.max_tokens = max_tokens

    @torch.no_grad()
    def generate_audio(
        self,
        audio,
        max_new_tokens: int = None,
    ) -> str:
        """
        Generate text response from audio input (dict format).

        Args:
            audio: Audio input dict with 'array' and 'sampling_rate' keys
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            str: Model's text response
        """
        # Ensure audio is at 16kHz
        if audio['sampling_rate'] != 16000:
            raise ValueError(f"Expected 16kHz audio, got {audio['sampling_rate']}Hz")

        # Save audio to temporary file
        import tempfile
        import soundfile as sf

        audio_array = audio['array']
        if isinstance(audio_array, np.ndarray):
            audio_array = audio_array.astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio_array, 16000)

        try:
            return self.generate_from_file(temp_path, max_new_tokens=max_new_tokens)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @torch.no_grad()
    def generate_text(
        self,
        text: str,
        max_new_tokens: int = None,
    ) -> str:
        """
        Generate response from text input.

        Args:
            text: Text input string
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            str: Model's text response
        """
        max_new_tokens = max_new_tokens or self.max_tokens

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "human", "content": text},
            {"role": "assistant", "content": None}
        ]

        _, response_text, _ = self.model(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=self.text_temperature,
            top_p=self.text_top_p,
            do_sample=True
        )

        return response_text

    @torch.no_grad()
    def generate_from_file(
        self,
        audio_path: str,
        max_new_tokens: int = None,
        with_audio_output: bool = False,
    ) -> Tuple[str, Optional[bytes]]:
        """
        Generate response from audio file path.

        Args:
            audio_path: Path to audio file
            max_new_tokens: Maximum number of tokens to generate
            with_audio_output: Whether to generate audio output

        Returns:
            If with_audio_output=False: str (text response)
            If with_audio_output=True: Tuple[str, bytes] (text, audio_bytes)
        """
        max_new_tokens = max_new_tokens or self.max_tokens

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "human", "content": [{"type": "audio", "audio": audio_path}]},
        ]

        if with_audio_output:
            messages.append({"role": "assistant", "content": "<tts_start>", "eot": False})

            tokens, text, audio_tokens = self.model(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=self.audio_temperature,
                top_p=self.audio_top_p,
                do_sample=True
            )

            # Convert audio tokens to wav
            audio_bytes = None
            if audio_tokens:
                audio_tokens = [x for x in audio_tokens if x < 6561]
                if audio_tokens:
                    audio_bytes = self.token2wav(audio_tokens, prompt_wav=self._prompt_wav)

            return text, audio_bytes
        else:
            messages.append({"role": "assistant", "content": None})

            _, text, _ = self.model(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=self.text_temperature,
                top_p=self.text_top_p,
                do_sample=True
            )

            return text, None

    @torch.no_grad()
    def generate_multi_round(
        self,
        audio_paths: List[str],
        with_audio_output: bool = True,
        max_new_tokens: int = None,
    ) -> List[Tuple[str, Optional[bytes]]]:
        """
        Generate multi-round conversation responses.

        Args:
            audio_paths: List of audio file paths for each round
            with_audio_output: Whether to generate audio output
            max_new_tokens: Maximum number of tokens to generate per round

        Returns:
            List of (text_response, audio_bytes) tuples for each round
        """
        max_new_tokens = max_new_tokens or self.max_tokens
        results = []

        # Initialize conversation history
        history = [{"role": "system", "content": self.system_prompt}]

        for round_idx, audio_path in enumerate(audio_paths, start=1):
            # Add user message
            history.append({
                "role": "human",
                "content": [{"type": "audio", "audio": audio_path}]
            })

            if with_audio_output:
                # Add assistant message for audio output
                history.append({
                    "role": "assistant",
                    "content": "<tts_start>",
                    "eot": False
                })

                tokens, text, audio_tokens = self.model(
                    history,
                    max_new_tokens=max_new_tokens,
                    temperature=self.audio_temperature,
                    top_p=self.audio_top_p,
                    do_sample=True
                )

                # Convert audio tokens to wav
                audio_bytes = None
                if audio_tokens:
                    audio_tokens = [x for x in audio_tokens if x < 6561]
                    if audio_tokens:
                        audio_bytes = self.token2wav(audio_tokens, prompt_wav=self._prompt_wav)

                # Update history with full response
                history.pop(-1)
                history.append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "<tts_start>"},
                        {"type": "token", "token": tokens}
                    ]
                })

                results.append((text, audio_bytes))
            else:
                # Text-only output
                history.append({
                    "role": "assistant",
                    "content": None
                })

                tokens, text, _ = self.model(
                    history,
                    max_new_tokens=max_new_tokens,
                    temperature=self.text_temperature,
                    top_p=self.text_top_p,
                    do_sample=True
                )

                # Update history
                history.pop(-1)
                history.append({
                    "role": "assistant",
                    "content": text
                })

                results.append((text, None))

        return results
