import os
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import torch

from vmem_bench.common.model_weights import weights_root
from .transnetv2_pytorch import TransNetV2 as PyTorchTransNetV2


class TransNetV2:
    """Wrapper class for TransNetV2 model loading, inference, and auto-downloading."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str = "cuda",
        threshold: float = 0.5,
    ) -> None:
        if weights_path is None:
            self.weights_path = (
                weights_root()
                / "shot_boundary_detection"
                / "TransNetV2"
                / "transnetv2-pytorch-weights.pth"
            )
        else:
            self.weights_path = Path(weights_path)

        self.device_str = device
        self.threshold = threshold

        if self.device_str == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self._ensure_weights_exist()

        self.model = PyTorchTransNetV2()
        state_dict = torch.load(self.weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

    def _ensure_weights_exist(self) -> None:
        """Check if weights exist locally; if not, download them from Hugging Face."""
        if not self.weights_path.exists():
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://huggingface.co/Sn4kehead/TransNetV2/resolve/main/transnetv2-pytorch-weights.pth"
            print(f"Downloading TransNetV2 weights from {url} to {self.weights_path}...")
            try:
                # Use urllib to download chunk-by-chunk with progress reporting
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                )
                with urllib.request.urlopen(req) as response, open(self.weights_path, "wb") as out_file:
                    meta = response.info()
                    file_size = int(meta.get("Content-Length", 0))
                    print(f"File size: {file_size / (1024 * 1024):.2f} MB")

                    block_size = 8192
                    downloaded = 0
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        downloaded += len(buffer)
                        out_file.write(buffer)
                        if file_size > 0:
                            percent = downloaded * 100 / file_size
                            if downloaded % (block_size * 100) == 0 or downloaded == file_size:
                                print(f"Downloaded: {percent:.1f}% ({downloaded / (1024 * 1024):.2f} MB)", end="\r")
                    print("\nDownload complete.")
            except Exception as e:
                if self.weights_path.exists():
                    self.weights_path.unlink()
                raise RuntimeError(f"Failed to download TransNetV2 weights: {e}") from e

    def predict_video(self, video_path: str | Path) -> tuple[np.ndarray, list[tuple[float, float]]]:
        """Predict scenes for a video.

        Returns:
            predictions: raw frame-level predictions.
            scenes: list of scene start and end times in seconds [(start_sec, end_sec), ...].
        """
        video_path = str(video_path)
        frames, fps, total_frames, frame_size = self._extract_frames(video_path)
        if len(frames) == 0:
            return np.array([]), []

        predictions = self._run_inference(frames)
        scenes_frames = self._predictions_to_scenes(predictions)

        # Convert frame indices to seconds
        scenes_seconds = []
        for start_frame, end_frame in scenes_frames:
            start_sec = start_frame / fps
            end_sec = end_frame / fps
            scenes_seconds.append((start_sec, end_sec))

        return predictions, scenes_seconds

    def _extract_frames(self, video_path: str, target_height: int = 27, target_width: int = 48):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (orig_width, orig_height)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (target_width, target_height))
            frames.append(frame_resized)

        cap.release()
        return np.array(frames), fps, total_frames, frame_size

    def _input_iterator(self, frames):
        no_padded_frames_start = 25
        no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)

        start_frame = np.expand_dims(frames[0], 0)
        end_frame = np.expand_dims(frames[-1], 0)
        padded_inputs = np.concatenate(
            [start_frame] * no_padded_frames_start +
            [frames] +
            [end_frame] * no_padded_frames_end, 0
        )

        ptr = 0
        while ptr + 100 <= len(padded_inputs):
            out = padded_inputs[ptr:ptr + 100]
            ptr += 50
            yield out[np.newaxis]

    def _run_inference(self, frames):
        with torch.no_grad():
            predictions = []
            for inp in self._input_iterator(frames):
                video_tensor = torch.from_numpy(inp).to(self.device)
                single_frame_pred, all_frame_pred = self.model(video_tensor)

                single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
                predictions.append(single_frame_pred[0, 25:75, 0])

            single_frame_pred = np.concatenate(predictions)
            return single_frame_pred[:len(frames)]

    def _predictions_to_scenes(self, predictions):
        predictions = (predictions > self.threshold).astype(np.uint8)
        scenes = []
        t_prev = 0
        start = 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t

        if t_prev == 0:
            scenes.append([start, len(predictions) - 1])

        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)
