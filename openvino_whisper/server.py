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
DEVICE = os.getenv("DEVICE", "GPU")

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
    
    # Environment tuning for 13th Gen GPU
    os.environ["OV_GPU_WAIT_FOR_DYNAMICS"] = "1"
    
    try:
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID, device=DEVICE, export=True, compile=True,
            ov_config={"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": ""}
        )
        current_dev = DEVICE
    except Exception as e:
        _LOGGER.error(f"GPU Load failed, falling back to CPU: {e}")
        model = OVModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID, device="CPU", export=True, compile=True
        )
        current_dev = "CPU"
    
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    pipe = pipeline(
        "automatic-speech-recognition", 
        model=model, 
        feature_extractor=processor.feature_extractor, 
        tokenizer=processor.tokenizer
    )
    
    attr = Attribution(name="OpenAI", url="https://github.com/openai/whisper")
    
    # USING KEYWORD ARGUMENTS TO PREVENT TYPEERROR
    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="OpenVINO Whisper",
                description="Intel OpenVINO accelerated Whisper STT",
                attribution=attr,
                version="8.0.1",
                models=[
                    AsrModel(
                        name=MODEL_ID,
                        description="Large Turbo Whisper",
                        attribution=attr,
                        version="1.0",
                        languages=["en"]
                    )
                ]
            )
        ]
    )
    
    server = AsyncServer.from_uri("tcp://0.0.0.0:10300")
    _LOGGER.info(f"Ready! Running on {current_dev}")
    await server.run(lambda: OpenVINOWhisperHandler(wyoming_info, pipe))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
