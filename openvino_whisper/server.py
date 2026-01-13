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
    def __init__(self, wyoming_info_event, pipe, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wyoming_info_event = wyoming_info_event
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
    
    model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, device=os.getenv("DEVICE"), export=True, compile=True)
    proc = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline("automatic-speech-recognition", model=model, feature_extractor=proc.feature_extractor, tokenizer=proc.tokenizer)

    wyoming_info = Info(asr=[AsrProgram(
        name="OpenVINO Whisper", description="Accelerated STT", attribution=Attribution(name="OpenAI", url=""),
        version="1.0.0", installed=True,
        models=[AsrModel(name=model_id, description="Turbo", attribution=Attribution(name="OpenAI", url=""),
                         version="1.0", languages=[os.getenv("LANGUAGE")], installed=True)]
    )])

    server = AsyncServer.from_uri("tcp://0.0.0.0:10300")
    await server.run(lambda: OpenVINOWhisperHandler(wyoming_info.event(), pipe))

if __name__ == "__main__":
    asyncio.run(main())
