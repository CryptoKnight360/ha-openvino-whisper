import asyncio
import logging
import os
import time
import numpy as np
from typing import Optional

from optimum.intel.openvino import OVSpeechSeq2SeqPipeline
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info, Model, ModelType, SttModel, SttProgram
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.stt import Transcribe, Transcription

# Configure Logging
logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

# Environment variables
MODEL_ID = os.getenv("MODEL_ID", "OpenVINO/whisper-small-int8-ov")
DEVICE = os.getenv("DEVICE", "AUTO")

class OpenVINOWhisperHandler(AsyncEventHandler):
    def __init__(self, wyoming_info: Info, pipe: OVSpeechSeq2SeqPipeline, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.pipe = pipe
        self.audio_buffer = bytearray()

    async def handle_event(self, event: Event) -> bool:
        if Transcribe.is_type(event.type):
            # Client wants to start transcribing
            pass
        
        elif AudioChunk.is_type(event.type):
            # Receive audio data
            chunk = AudioChunk.from_event(event)
            self.audio_buffer.extend(chunk.audio)

        elif AudioStop.is_type(event.type):
            # End of audio stream - perform inference
            _LOGGER.info("Audio received, transcribing...")
            start_time = time.perf_counter()
            
            # Convert buffer to float32 numpy array (normalized)
            audio_array = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Run OpenVINO Inference
            # Whisper expects 16000Hz. Wyoming protocol usually sends 16000Hz.
            result = self.pipe(audio_array, sampling_rate=16000)
            text = result["text"].strip()
            
            inference_ms = (time.perf_counter() - start_time) * 1000
            _LOGGER.info(f"Transcription: {text} (Latency: {inference_ms:.1f}ms)")

            # Send result back to Wyoming client
            await self.write_event(Transcription(text=text).event())
            
            # Return False to close the connection after one transcription
            return False

        elif Describe.is_type(event.type):
            await self.write_event(self.wyoming_info.event())

        return True

async def main():
    _LOGGER.info(f"Loading model {MODEL_ID} to {DEVICE}...")
    
    # Load model once at startup
    pipe = OVSpeechSeq2SeqPipeline.from_pretrained(
        MODEL_ID,
        device=DEVICE,
        compile=True
    )

    # Define Wyoming Info for Home Assistant discovery
    wyoming_info = Info(
        stt=[
            SttProgram(
                name="OpenVINO Whisper",
                slug="openvino_whisper",
                description="Intel OpenVINO accelerated Whisper STT",
                version="2.0.0",
                models=[
                    SttModel(
                        name=MODEL_ID,
                        slug=MODEL_ID,
                        description="Quantized Whisper",
                        attribution={"name": "Intel", "url": "https://github.com/openvinotoolkit/openvino"},
                        installed=True,
                        languages=["en"], # Add more as needed
                        version="1.0"
                    )
                ]
            )
        ]
    )

    server = AsyncServer.from_uri("tcp://0.0.0.0:10300")
    _LOGGER.info("Ready! Listening on port 10300")
    
    await server.run(lambda: OpenVINOWhisperHandler(wyoming_info, pipe))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
