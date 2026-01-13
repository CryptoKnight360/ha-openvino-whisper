import asyncio
import logging
import os
import time
import numpy as np
from optimum.intel.openvino import OVSpeechSeq2SeqPipeline
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info, SttModel, SttProgram
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.stt import Transcribe, Transcription

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

# These are pulled from your Home Assistant Configuration tab
MODEL_ID = os.getenv("MODEL_ID", "openai/whisper-large-v3-turbo")
DEVICE = os.getenv("DEVICE", "AUTO")

class OpenVINOWhisperHandler(AsyncEventHandler):
    def __init__(self, wyoming_info: Info, pipe: OVSpeechSeq2SeqPipeline, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.pipe = pipe
        self.audio_buffer = bytearray()

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self.audio_buffer.extend(chunk.audio)
        elif AudioStop.is_type(event.type):
            _LOGGER.info("Transcribing with Large Turbo...")
            start_time = time.perf_counter()
            audio_array = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Inference
            result = self.pipe(audio_array, sampling_rate=16000)
            text = result["text"].strip()
            
            _LOGGER.info(f"Transcription: {text} ({(time.perf_counter() - start_time)*1000:.1f}ms)")
            await self.write_event(Transcription(text=text).event())
            return False
        elif Describe.is_type(event.type):
            await self.write_event(self.wyoming_info.event())
        return True

async def main():
    _LOGGER.info(f"Loading {MODEL_ID} to {DEVICE} (Exporting if needed)...")
    
    # export=True allows it to download and convert the Large Turbo model automatically
    pipe = OVSpeechSeq2SeqPipeline.from_pretrained(
        MODEL_ID,
        device=DEVICE,
        export=True, 
        compile=True,
        load_in_8bit=False # Use False for better accuracy on Large Turbo
    )

    wyoming_info = Info(stt=[SttProgram(name="OpenVINO Whisper", slug="openvino_whisper", description="OpenVINO Whisper", version="2.1.0", models=[SttModel(name=MODEL_ID, slug=MODEL_ID, description="Whisper Model", installed=True, languages=["en"])])])
    server = AsyncServer.from_uri("tcp://0.0.0.0:10300")
    _LOGGER.info("Ready!")
    await server.run(lambda: OpenVINOWhisperHandler(wyoming_info, pipe))

if __name__ == "__main__":
    asyncio.run(main())
