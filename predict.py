# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
import subprocess
import random
import math
import json
import whisperx
import torch
from cog import BasePredictor, Input, Path
os.environ['HF_HOME'] = '/src/hf_models'
os.environ['TORCH_HOME'] = '/src/torch_models'


compute_type = "float16"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda"
        self.model = whisperx.load_model(
            "large-v3", self.device, compute_type=compute_type)
        self.allign_model_en, self.metadata_en = whisperx.load_align_model(language_code='en', device=self.device)
        self.allign_model_ru, self.metadata_ru = whisperx.load_align_model(language_code='ru', device=self.device)

    def predict(
        self,
        audio: Path = Input(description="Audio file", default="https://pyannote-speaker-diarization.s3.eu-west-2.amazonaws.com/lex-levin-4min.mp3"),
        batch_size: int = Input(description="Parallelization of input audio transcription", default=16),
        hugging_face_token: str = Input(description="Your Hugging Face access token. If empty skip diarization.", default=None),
        language: str = Input(description="Spoken language. If empty auto detect.", default=None),
        debug: bool = Input(description="Print out memory usage information.", default=True)
    ) -> str:
        self.file_path = str(audio)
        """Run a single prediction on the model"""
        with torch.inference_mode():
            try:
                # 0: Detect lang
                language = self.detect_lang_from_several_parts()

                # 1. Transcribe with original whisper (batched)                
                audio = whisperx.load_audio(self.file_path)
                result = self.model.transcribe(audio, batch_size=batch_size, language=language)

                # 2. Align whisper output
                lang = result["language"]
                if lang == 'en':
                    result = whisperx.align(result['segments'], self.allign_model_en, self.metadata_en, audio, self.device, return_char_alignments=False)
                elif lang == 'ru':
                    result = whisperx.align(result['segments'], self.allign_model_ru, self.metadata_ru, audio, self.device, return_char_alignments=False)
                elif lang != 'nn':
                    try:
                        model_a, metadata = whisperx.load_align_model(language_code=lang, device=self.device)
                        result = whisperx.align(result['segments'], model_a, metadata, audio, self.device, return_char_alignments=False)
                    except Exception as e:
                        print(e)

                # 3. Assign speaker labels
                if hugging_face_token:
                    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hugging_face_token, device=self.device)
                    diarize_segments = diarize_model(audio)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as e:
                return json.dumps(f"Error occurred {e}")

            if debug:
                print(
                    f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")
        return json.dumps([result['segments'], lang])

    def detect_lang_from_several_parts(self):
        duration_sec = math.floor(self.get_audio_duration())
        if duration_sec <= 30:
            return None

        interval_langs = []
        # Split into 30s intervals 
        all_30sec_intervals = self.calculate_time_intervals(duration_sec)
        intervals = random.choices(all_30sec_intervals, k=4)
        for idx, interval in enumerate(intervals):
            cut_interval_path = self.cut_recording(idx, interval[0])
            res = self.detect_chunk_lang(cut_interval_path)
            interval_langs.append(res)
            os.remove(cut_interval_path)

        return max(set(interval_langs), key=interval_langs.count)

    def detect_chunk_lang(self):
        self.read_audio()
        lang = self.get_language()
        return lang
    
    def read_audio(self):
        audio = whisperx.load_audio(self.file_path)
        audio = whisperx.pad_or_trim(audio)
        # make log-Mel spectrogram and move to the same device as the model
        self.mel = whisperx.log_mel_spectrogram(audio).to(self.device)

    def get_language(self):
        _, probs = self.model.detect_language(self.mel)
        return max(probs, key=probs.get)

    def calculate_time_intervals(self, duration_sec):
        intervals = []
        if duration_sec < 30:
            return intervals
        start = 0
        while start + 30 <= duration_sec:
            intervals.append((start, start + 30))
            start += 30
        return intervals
    
    def cut_recording(self, chunk_postfix, from_sec):
        file_pathname, ext = self.get_file_name_and_ext()
        cut_file_path = f"{file_pathname}_{chunk_postfix}{ext}"
        subprocess.call(['ffmpeg', '-loglevel', 'error', '-i', self.file_path,
                        '-ss', f'{from_sec}', '-t', '30', '-map', '0', '-c', 'copy', cut_file_path])
        return cut_file_path
    
    def get_file_name_and_ext(self):
        return os.path.splitext(self.file_path)
    
    def get_audio_duration(self):
        result = subprocess.run(["ffprobe", "-loglevel", "error", "-show_entries",
                                "format=duration", "-of",
                                "default=noprint_wrappers=1:nokey=1", self.file_path],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        return float(result.stdout)
