import asyncio
import logging
import os
import numpy as np
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor, pipeline
from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStop, AudioStart
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

_LOGGER = logging.getLogger(__name__)

class OpenVINOWhisperHandler(AsyncEventHandler):
    def __init__(self, wyoming_info: Info, pipe: pipeline, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wyoming_info_event = wyoming_info.event()
        self.pipe = pipe
        self.audio_buffer = bytearray()

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True
        if AudioStart.is_type(event.type):
            self.audio_buffer = bytearray()
            return True
        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self.audio_buffer.extend(chunk.audio)
            return True
        if AudioStop.is_type(event.type):
            audio_array = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            result = await asyncio.to_thread(self.pipe, audio_array)
            await self.write_event(Transcript(text=result["text"].strip()).event())
            return False
        return True

async def main():
    logging.basicConfig(level=logging.INFO)
    model_id = os.getenv("MODEL_ID")
    device = os.getenv("DEVICE")
    
    _LOGGER.info(f"Loading {model_id} on {device}...")
    model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, device=device, export=True, compile=True)
    proc = AutoProcessor.from_pretrained(model_id)
    
    # Use feature_extractor (NOT extractor) to match Transformers API
    pipe = pipeline("automatic-speech-recognition", model=model, feature_extractor=proc.feature_extractor, tokenizer=proc.tokenizer)

    attr = Attribution(name="OpenAI", url="https://openai.com")
    wyoming_info = Info(asr=[AsrProgram(
        name="openvino-whisper", description="OpenVINO STT", attribution=attr, version="1.0.0", installed=True,
        models=[AsrModel(name="whisper-large-v3-turbo", languages=["en"], attribution=attr, installed=True, version="1.0.0")]
    )])

    server = AsyncServer.from_uri("tcp://0.0.0.0:10300")
    _LOGGER.info("Ready and listening on port 10300")
    await server.run(lambda: OpenVINOWhisperHandler(wyoming_info, pipe))

if __name__ == "__main__":
    asyncio.run(main())
