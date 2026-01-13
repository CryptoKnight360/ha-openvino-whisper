import asyncio
import logging
import os
import time
import numpy as np
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor, pipeline
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

logging.basicConfig(level=logging.INFO)
_LOGGER = logging.getLogger(__name__)

MODEL_ID = os.getenv("MODEL_ID", "openai/whisper-large-v3-turbo")
DEVICE = "CPU"

class OpenVINOWhisperHandler(AsyncEventHandler):
    def __init__(self, wyoming_info: Info, pipe: pipeline, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.pipe = pipe
        self.audio_buffer = bytearray()

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info.event())
            return True
        if AudioChunk.is_type(event.type):
            self.audio_buffer.extend(AudioChunk.from_event(event).audio)
            return True
        if AudioStop.is_type(event.type):
            start = time.perf_counter()
            audio = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            result = self.pipe(audio)
            _LOGGER.info(f"Text: {result['text']} ({(time.perf_counter() - start)*1000:.0f}ms)")
            await self.write_event(Transcript(text=result['text'].strip()).event())
            return False
        return True

async def main():
    _LOGGER.info(f"Loading {MODEL_ID}...")
    model = OVModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, device=DEVICE, export=True, compile=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    pipe = pipeline("automatic-speech-recognition", model=model, feature_extractor=proc.feature_extractor, tokenizer=proc.tokenizer)
    
    attr = Attribution(name="OpenAI", url="https://github.com/openai/whisper")
    
    # Corrected: Added 'installed=True' to both classes
    wyoming_info = Info(asr=[AsrProgram(
        name="OpenVINO Whisper",
        description="Whisper STT",
        attribution=attr,
        version="15.0.0",
        installed=True,
        models=[AsrModel(
            name=MODEL_ID, 
            description="Turbo", 
            attribution=attr, 
            version="1.0", 
            languages=["en"],
            installed=True
        )]
    )])

    server = AsyncServer.from_uri("tcp://0.0.0.0:10300")
    _LOGGER.info("Ready!")
    await server.run(lambda: OpenVINOWhisperHandler(wyoming_info, pipe))

if __name__ == "__main__":
    asyncio.run(main())
