# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
import subprocess
import random
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
        self.allign_model_ru, self.metadata_ru = whisperx.load_align_model(language_code='ru', device=self.device)

    def predict(
        self,
        audio: Path = Input(description="Audio file", default="https://pyannote-speaker-diarization.s3.eu-west-2.amazonaws.com/lex-levin-4min.mp3"),
        batch_size: int = Input(description="Parallelization of input audio transcription", default=32),
        hugging_face_token: str = Input(description="Your Hugging Face access token. If empty skip diarization.", default=None),
        duration_sec: int = Input(description="Duration in sec to improve lang detection", default=0),
        debug: bool = Input(description="Print out memory usage information.", default=True),
        lang: str = Input(description="Predefined lang", default=None),
    ) -> str:
        self.file_path = str(audio)
        self.duration_sec = duration_sec
        """Run a single prediction on the model"""
        with torch.inference_mode():
            try:
                if lang:
                    language = lang
                else:
                    # 0: Detect lang
                    print("Start lang detection")
                    language = self.detect_lang_from_several_parts()

                # 1. Transcribe with original whisper (batched)
                audio = whisperx.load_audio(self.file_path)
                result = self.model.transcribe(audio, batch_size=batch_size, language=language)

                # 2. Align whisper output
                lang = result["language"]
                if lang == 'ru':
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
        duration_sec = self.duration_sec
        if duration_sec <= 30:
            return None

        interval_langs = []
        # Split into 31s intervals 
        all_31sec_intervals = self.calculate_time_intervals(duration_sec)
        intervals = random.sample(all_31sec_intervals, min(len(all_31sec_intervals), 5))
        for idx, interval in enumerate(intervals):
            print(f"Detect lang for {interval}")
            cut_interval_path = self.cut_recording(idx, interval[0])
            cut_audio = whisperx.load_audio(cut_interval_path)
            res = self.model.detect_language(cut_audio)
            if res != "nn":
                interval_langs.append(res)
            print(f"-- {res}")
            os.remove(cut_interval_path)

        print(f"Detected langs: {interval_langs}")
        return max(set(interval_langs), key=interval_langs.count)

    def calculate_time_intervals(self, duration_sec):
        intervals = []
        if duration_sec < 30:
            return intervals
        start = 0
        while start + 31 <= duration_sec:
            intervals.append((start, start + 31))
            start += 31
        return intervals
    
    def cut_recording(self, chunk_postfix, from_sec):
        file_pathname, ext = self.get_file_name_and_ext()
        cut_file_path = f"{file_pathname}_{chunk_postfix}{ext}"
        subprocess.call(['ffmpeg', '-loglevel', 'error', '-i', self.file_path,
                        '-ss', f'{from_sec}', '-t', '31', '-map', '0', '-c', 'copy', cut_file_path])
        return cut_file_path
    
    def get_file_name_and_ext(self):
        return os.path.splitext(self.file_path)
