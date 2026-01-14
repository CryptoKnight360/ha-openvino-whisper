import argparse
import asyncio
import logging
import socket
import numpy as np
import time
import os

from wyoming.server import AsyncServer, AsyncEventHandler
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.info import Describe, Info, AsrProgram, AsrModel, Attribution
from wyoming.asr import Transcribe, Transcript
from wyoming.event import Event

from optimum.intel import OVModelForSpeechSeq2Seq
from transformers import AutoTokenizer, pipeline
from zeroconf import ServiceInfo, Zeroconf

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class OpenVINOWhisperHandler(AsyncEventHandler):
    def __init__(self, cli_args, pipe, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.pipe = pipe
        self.audio_bytes = b""
        self.is_transcribing = False

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(
                Info(
                    asr=[
                        AsrProgram(
                            name="openvino-whisper",
                            description="Whisper ASR accelerated by OpenVINO",
                            attribution=Attribution(
                                name="Intel OpenVINO",
                                url="https://github.com/huggingface/optimum-intel",
                            ),
                            installed=True,
                            models=[
                                AsrModel(
                                    name=self.cli_args.model,
                                    description="Selected OpenVINO Model",
                                    attribution=Attribution(
                                        name="HuggingFace",
                                        url=f"https://huggingface.co/{self.cli_args.model}",
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

        if AudioStart.is_type(event.type):
            self.audio_bytes = b""
            self.is_transcribing = True
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self.audio_bytes += chunk.audio
            return True

        if AudioStop.is_type(event.type):
            self.is_transcribing = False
            logger.info("Audio captured: %s bytes. Starting inference...", len(self.audio_bytes))
            
            start_time = time.time()
            text = await asyncio.to_thread(self._transcribe, self.audio_bytes)
            end_time = time.time()
            
            logger.info("Inference done in %.2fs. Transcript: '%s'", end_time - start_time, text)
            await self.write_event(Transcript(text=text).event())
            return False

        return True

    def _transcribe(self, audio_data: bytes) -> str:
        # Convert 16-bit PCM to float32
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        try:
            result = self.pipe(
                audio_array, 
                generate_kwargs={
                    "language": self.cli_args.language, 
                    "num_beams": self.cli_args.beam_size,
                    "max_new_tokens": 128 
                }
            )
            return result["text"].strip()
        except Exception as e:
            logger.error("Inference failed: %s", e)
            return ""

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai/whisper-large-v3-turbo")
    parser.add_argument("--device", type=str, default="GPU")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--port", type=int, default=10300)
    args = parser.parse_args()

    # Cache Directory Setup
    cache_dir = "/data/model_cache"
    os.makedirs(cache_dir, exist_ok=True)

    logger.info("----------------------------------------------------------------")
    logger.info("Wyoming OpenVINO Whisper (LattePanda Optimized)")
    logger.info("Model: %s", args.model)
    logger.info("Device: %s", args.device)
    logger.info("----------------------------------------------------------------")
    
    if "large" in args.model:
        logger.warning("NOTE: Large/Turbo models take time to convert on first run.")

    try:
        # Load Model
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
            args.model, 
            device=args.device, 
            export=True,
            compile=True, 
            ov_config={"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": cache_dir}
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=ov_model,
            tokenizer=tokenizer,
            feature_extractor=tokenizer.feature_extractor,
            chunk_length_s=30,
        )
        
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.critical("Failed to load model: %s", e)
        return

    server = AsyncServer.fromk_uri(f"tcp://0.0.0.0:{args.port}")
    
    # Zeroconf Discovery
    zeroconf = Zeroconf()
    info = ServiceInfo(
        "_wyoming._tcp.local.",
        "Wyoming OpenVINO Whisper._wyoming._tcp.local.",
        addresses=[socket.inet_aton("0.0.0.0")],
        port=args.port,
        properties={"host": socket.gethostname(), "mac": "00:00:00:00:00:00"},
    )
    zeroconf.register_service(info)
    
    logger.info("Wyoming Server ready on port %s", args.port)

    try:
        await server.run(
            lambda *args_svr, **kwargs_svr: OpenVINOWhisperHandler(args, asr_pipe, *args_svr, **kwargs_svr)
        )
    except KeyboardInterrupt:
        pass
    finally:
        zeroconf.unregister_service(info)
        zeroconf.close()

if __name__ == "__main__":
    asyncio.run(main())
