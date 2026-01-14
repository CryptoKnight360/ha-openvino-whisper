import asyncio
import numpy as np
import openvino_genai as ov_genai
from wyoming.server import AsyncServer
from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event

class OpenVINOWhisperServer(AsyncServer):
    def __init__(self, model_id, device, host, port):
        super().__init__(host, port)
        print(f"Loading {model_id} on {device}...")
        self.pipe = ov_genai.WhisperPipeline(model_id, device)
        self.audio_data = []

    async def handle_event(self, event: Event, stdin, stdout):
        if isinstance(event, AudioChunk):
            chunk = np.frombuffer(event.audio, dtype=np.int16).astype(np.float32) / 32768.0
            self.audio_data.append(chunk)
        elif isinstance(event, AudioStop):
            if not self.audio_data:
                return True
            full_audio = np.concatenate(self.audio_data)
            result = self.pipe.generate(full_audio)
            await self.write_event(Transcript(text=result.texts[0]), stdout)
            self.audio_data = [] 
        return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="CPU")
    args = parser.parse_args()
    server = OpenVINOWhisperServer(args.model, args.device, "0.0.0.0", 10300)
    asyncio.run(server.run())
