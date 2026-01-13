import time
import numpy as np
import webrtcvad
from wyoming.server import AsyncServer
from wyoming.stt import SpeechToText
from optimum.intel.openvino import OVSpeechSeq2SeqPipeline
from transformers import AutoProcessor

MODEL_NAME = "OpenVINO/whisper-small-int8-ov"
DEVICE = "AUTO"

class OpenVINOWhisperSTT(SpeechToText):
    def __init__(self):
        self.vad = webrtcvad.Vad(2)
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.pipe = OVSpeechSeq2SeqPipeline.from_pretrained(
            MODEL_NAME,
            device=DEVICE
        )

    def apply_vad(self, audio, rate):
        frame_len = int(rate * 0.02)
        voiced = []
        for i in range(0, len(audio), frame_len):
            frame = audio[i:i+frame_len]
            if len(frame) < frame_len:
                continue
            pcm = (frame * 32768).astype(np.int16).tobytes()
            if self.vad.is_speech(pcm, rate):
                voiced.extend(frame)
        return np.array(voiced, dtype=np.float32)

    async def transcribe(self, audio: bytes, rate: int):
        start = time.time()
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        samples = self.apply_vad(samples, rate)

        if len(samples) == 0:
            return ""

        result = self.pipe(samples, sampling_rate=rate)
        text = result["text"].strip()

        latency = (time.time() - start) * 1000
        print(f"STT latency: {latency:.1f} ms")

        return text

async def main():
    server = AsyncServer.from_uri("tcp://0.0.0.0:10300")
    stt = OpenVINOWhisperSTT()
    await server.run(stt)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
