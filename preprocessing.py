import cv2
import numpy as np

#test frame extraction and preprocessing


def preprocess(video_path, resize=(224, 224), max_frames=50):
    video_path = video_path.decode("utf-8") if isinstance(video_path, bytes) else video_path
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = 7
    start_frame, end_frame = 0, total_frames
    frames = []
    display_frames = []
    empty_frame = np.zeros((resize[1], resize[0], 3))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while len(frames) < max_frames and cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (cap.get(cv2.CAP_PROP_POS_FRAMES - 1 - start_frame) % interval) == 0:
            original_frame = frame.copy()
            frame = cv2.resize(frame, resize)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
            frame = frame.astype(np.float32) / 255.0
            frame_display = cv2.resize(original_frame, resize)
            frame_display = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            display_frames.append(frame_display)

    cap.release()
    # Pad frames if fewer than max_frames were captured
    if len(frames) < max_frames:
        i = 0
        while len(frames) < max_frames:
            frames.append(frames[i])
            display_frames.append(display_frames[i])
            i += 1
    frames_to_prediction = np.expand_dims(frames, axis=0)
    return np.array(frames_to_prediction, dtype=np.float32), np.array(display_frames, dtype=np.float32)
