import argparse
import asyncio
import logging
import sys
import tempfile
import wave
from pathlib import Path
from typing import Optional

import torch
from wyoming.event import Event
from wyoming.info import Describe, Info, AsrProgram, AsrModel, Attrib
from wyoming.server import AsyncServer, AsyncEventHandler
from wyoming.audio import AudioChunk, AudioStart, AudioStop

from transformers import AutoProcessor
from optimum.intel.openvino import OVModelForSpeechSeq2Seq

_LOGGER = logging.getLogger(__name__)

class State:
    """Internal state for the Whisper event handler."""
    def __init__(self):
        self.audio_bytes = b""
        self.is_receiving = False

class OpenVINOEventHandler(AsyncEventHandler):
    def __init__(
        self,
        cli_args: argparse.Namespace,
        model: OVModelForSpeechSeq2Seq,
        processor: AutoProcessor,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.model = model
        self.processor = processor
        self.state = State()

    async def handle_event(self, event: Event) -> bool:
        if AudioStart.is_type(event.type):
            self.state.is_receiving = True
            self.state.audio_bytes = b""
            return True

        if AudioChunk.is_type(event.type):
            if self.state.is_receiving:
                chunk = AudioChunk.from_event(event)
                self.state.audio_bytes += chunk.audio
            return True

        if AudioStop.is_type(event.type):
            self.state.is_receiving = False
            await self._transcribe()
            return True

        if Describe.is_type(event.type):
            await self.write_event(
                Info(
                    asr=[
                        AsrProgram(
                            name="OpenVINO Whisper",
                            description="Intel OpenVINO accelerated Whisper",
                            attribution=Attrib(
                                name="CryptoKnight360",
                                url="https://github.com/CryptoKnight360/ha-openvino-whisper",
                            ),
                            installed=True,
                            models=[
                                AsrModel(
                                    name=self.cli_args.model,
                                    description=self.cli_args.model,
                                    attribution=Attrib(
                                        name="OpenAI",
                                        url="https://github.com/openai/whisper",
                                    ),
                                    installed=True,
                                    languages=[self.cli_args.language],
                                )
                            ],
                        )
                    ]
                ).event()
            )
            return True

        return True

    async def _transcribe(self):
        if not self.state.audio_bytes:
            return

        _LOGGER.debug("Processing %s bytes of audio", len(self.state.audio_bytes))

        # Save to temp wav file for processing (simplest way to handle headers/resampling)
        # Note: Wyoming sends raw 16-bit 16khz PCM mono usually.
        with tempfile.NamedTemporaryFile(suffix=".wav", mode="wb") as temp_wav:
            with wave.open(temp_wav.name, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)
                wav_file.writeframes(self.state.audio_bytes)
            
            # Read back using soundfile or similar handled by transformers processor
            # Actually, we can feed raw bytes if we structure it right, but file is safer for ffmpeg backend.
            
            try:
                # Use asyncio.to_thread for blocking inference
                text = await asyncio.to_thread(
                    self._run_inference, temp_wav.name
                )
                
                _LOGGER.info("Transcription: %s", text)
                
                # Send back transcript
                await self.write_event(
                    Event(
                        type="transcript",
                        data={"text": text, "context": {}}
                    )
                )
            except Exception as e:
                _LOGGER.error("Inference failed: %s", e)

    def _run_inference(self, audio_path: str) -> str:
        # Pre-process audio
        # Note: The processor handles loading the audio file via ffmpeg internally
        input_features = self.processor(
            audio_path, 
            return_tensors="pt", 
            sampling_rate=16000
        ).input_features.to(self.model.device)

        # Generate token ids
        predicted_ids = self.model.generate(
            input_features, 
            language=self.cli_args.language,
            num_beams=self.cli_args.beam_size
        )

        # Decode token ids to text
        transcription = self.processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription.strip()

async def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--language", default="en")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--uri", default="tcp://0.0.0.0:10300")
    args = parser.parse_args()

    _LOGGER.info(f"Loading model {args.model} on {args.device}...")
    
    # Load Model (Optimized for OpenVINO)
    # export=True forces conversion to IR format if not already present in cache
    model = OVModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        device=args.device.upper(),
        export=True, 
        compile=True
    )
    
    processor = AutoProcessor.from_pretrained(args.model)

    _LOGGER.info("Model loaded. Starting Wyoming server on %s", args.uri)

    server = AsyncServer.from_uri(args.uri)
    
    try:
        await server.run(
            lambda *args_h, **kwargs_h: OpenVINOEventHandler(
                args, model, processor, *args_h, **kwargs_h
            )
        )
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    asyncio.run(main())
