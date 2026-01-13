import asyncio
import logging
import os
import time
import numpy as np

# Optimum Intel & Transformers
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor, pipeline

# Wyoming Protocol
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

MODEL_ID = os.getenv("MODEL_ID", "openai/whisper-large-v3-turbo")
DEVICE = os.getenv("DEVICE", "AUTO")

class OpenVINOWhisperHandler(AsyncEventHandler):
    def __init__(self, wyoming_info: Info, pipe: pipeline, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.pipe = pipe
        self.audio_buffer = bytearray()

    async def handle_event(self, event: Event) -> bool:
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self.audio_buffer.extend(chunk.audio)
        elif AudioStop.is_type(event.type):
            start_time = time.perf_counter()
            audio_array = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            
            result = self.pipe(audio_array)
            text = result["text"].strip()
            
            _LOGGER.info(f"Transcription: {text} ({(time.perf_counter() - start_time)*1000:.1f}ms)")
            await self.write_event(Transcript(text=text).event())
            return False
        elif Describe.is_type(event.type):
            await self.write_event(self.wyoming_info.event())
        return True

async def main():
    _LOGGER.info(f"Loading {MODEL_ID} to {DEVICE}...")
    
    model = OVModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, device=DEVICE, export=True, compile=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    pipe = pipeline("automatic-speech-recognition", model=model, feature_extractor=processor.feature_extractor, tokenizer=processor.tokenizer)
    
    # Define attribution
    attr = Attribution(name="OpenAI", url="https://github.com/openai/whisper")
    
    # Metadata verified: Attribution and Version are REQUIRED
    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="OpenVINO Whisper",
                description="Intel OpenVINO accelerated Whisper STT",
                attribution=attr,
                installed=True,
                version="3.2.0",
                models=[
                    AsrModel(
                        name=MODEL_ID,
                        description="Large Turbo Whisper",
                        attribution=attr,
                        installed=True,
                        languages=["en"],
                        version="1.0"
                    )
                ]
            )
        ]
    )
    
    server = AsyncServer.from_uri("tcp://0.0.0.0:10300")
    _LOGGER.info("Ready! Listening on 10300")
    await server.run(lambda: OpenVINOWhisperHandler(wyoming_info, pipe))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
