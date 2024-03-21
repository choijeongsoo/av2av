import os
import soundfile as sf
import cv2
import ffmpeg

def process_units(units, reduce=False):
    if not reduce:
        return units

    out = [u for i, u in enumerate(units) if i == 0 or u != units[i - 1]]
    return out

def save_unit(unit, unit_path):
    os.makedirs(os.path.dirname(unit_path), exist_ok=True)
    with open(unit_path, "w") as f:
        f.write(unit)

def save_audio(audio, audio_path, sampling_rate=16000):
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    sf.write(
        audio_path,
        audio,
        sampling_rate,
    )

def extract_audio_from_video(video_path, save_audio_path, sampling_rate=16000):
    os.makedirs(os.path.dirname(save_audio_path), exist_ok=True)
    (
        ffmpeg.input(video_path)
        .output(
            save_audio_path,
            acodec="pcm_s16le",
            ac=1,
            ar=sampling_rate,
            loglevel="panic",
        )
        .run(overwrite_output=True)
    )

def save_video(audio, video, full_video, bbox, save_video_path, sampling_rate=16000, fps=25, vcodec="libx264"):
    os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
    temp_audio_path = os.path.splitext(save_video_path)[0]+".temp.wav"
    temp_video_path = os.path.splitext(save_video_path)[0]+".temp.avi"

    save_audio(audio, temp_audio_path, sampling_rate)

    frame_h, frame_w = full_video.shape[1], full_video.shape[2]
    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
    
    for p, f, c in zip(video, full_video, bbox):
        x1, y1, x2, y2 = [max(int(_), 0) for _ in c]
        p = cv2.resize(p, (x2 - x1, y2 - y1))
        try:
            f[y1:y2, x1:x2] = p
        except:
            height, width, c = f[y1:y2, x1:x2].shape
            p = cv2.resize(p, (width, height))
            f[y1:y2, x1:x2] = p
        out.write(f)

    out.release()

    ffmpeg.output(
        ffmpeg.input(temp_video_path),
        ffmpeg.input(temp_audio_path),
        save_video_path,
        vcodec="libx264",
        acodec="aac",
        loglevel="panic",
    ).run(overwrite_output=True)

    os.remove(temp_audio_path)
    os.remove(temp_video_path)

