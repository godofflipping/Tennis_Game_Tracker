import cv2
from torch import nn, load, no_grad
import torchvision.models as models
import torchvision.transforms as tfs


class KeypointsDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 14 * 2)
        self.model.load_state_dict(load(model_path, map_location='cpu'))
        self.transform = tfs.Compose([
            tfs.ToPILImage(),
            tfs.Resize((224, 224)),
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        # Because camera is not moving, we can predict only on the first image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        # unsqueeze - put another list on top of the image
        # squeeze - removes upper list from the image
        image = self.transform(image).unsqueeze(0)

        with no_grad():
            outputs = self.model(image)

        keypoints = outputs.squeeze().cpu().numpy()
        keypoints[::2] *= width / 224.0
        keypoints[1::2] *= height / 224.0

        return keypoints

    def draw_keypoints_frame(self, frame, keypoints):
        # run for all x coordinates
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])
            cv2.putText(frame, str(i // 2), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), cv2.FILLED)
        return frame

    def draw_keypoints_frames(self, frames, keypoints):
        output_frames = []
        for frame in frames:
            frame = self.draw_keypoints_frame(frame, keypoints)
            output_frames.append(frame)
        return output_frames
