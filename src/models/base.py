import torch
import time


class VoiceAssistant:
    """Base class for voice assistant models."""

    @torch.no_grad()
    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ):
        """
        Generate response from audio input.

        Args:
            audio: Audio input (dict with 'array' and 'sampling_rate')
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            str: Model's text response
        """
        raise NotImplementedError

    @torch.no_grad()
    def generate_text(
        self,
        text,
    ):
        """
        Generate response from text input.

        Args:
            text: Text input string

        Returns:
            str: Model's text response
        """
        raise NotImplementedError

    @torch.no_grad()
    def generate_ttft(
        self,
        audio,
    ):
        """
        Measure time to first token.

        Args:
            audio: Audio input

        Returns:
            float: Time to first token in seconds
        """
        tmp = time.perf_counter()
        self.generate_audio(audio, max_new_tokens=1)
        return time.perf_counter() - tmp
