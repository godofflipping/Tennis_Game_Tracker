def get_distance(point_1, point_2):
    return ((point_1[0] - point_2[0]) ** 2 +
            (point_1[1] - point_2[1]) ** 2) ** 0.5


def get_bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    x_center = int((x1 + x2) / 2)
    y_center = int((y1 + y2) / 2)
    return x_center, y_center


def get_foot_position(bbox):
    x1, _, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)


def get_bbox_height(bbox):
    return bbox[3] - bbox[1]


def get_xy_distance(point_1, point_2):
    return abs(point_1[0] - point_2[0]), \
           abs(point_1[1] - point_2[1])


def get_closest_keypoint(point, keypoints, keypoint_ids):
    min_distance = float('inf')
    index = keypoint_ids[0]
    for keypoint_id in keypoint_ids:
        keypoint = keypoints[keypoint_id * 2], keypoints[keypoint_id * 2 + 1]
        distance = abs(point[1] - keypoint[1])

        if distance < min_distance:
            min_distance = distance
            index = keypoint_id

    return index
