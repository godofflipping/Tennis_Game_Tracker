import cv2


def read_video(video_path):
    capture = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)
    capture.release()
    return frames


def draw_frame_numbers(frames):
    output_frames = []
    for i, frame in enumerate(frames):
        x, y = frame.shape[:2]
        cv2.putText(frame, f"Frame: {i + 1}", (int(y * 0.9), int(x * 0.075)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        output_frames.append(frame)
    return output_frames


def save_video(output_frames, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output = cv2.VideoWriter(
        output_path, fourcc, 24,
        (output_frames[0].shape[1], output_frames[0].shape[0])
    )
    for frame in output_frames:
        output.write(frame)
    output.release()
