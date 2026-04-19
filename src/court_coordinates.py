import cv2
import numpy as np


class KeypointSelector:
    """
    Manual keypoint selector for homography. User clicks 4 points on the video frame

    Rules:
    - User must click in the 4 corners of the court with the following order:
        1. Top-left corner
        2. Top-right corner
        3. Bottom-left corner
        4. Bottom-right corner
    """
    def __init__(self, video_path, missing_corner, display_width=1280, display_height=720):
        self.video_path = video_path
        self.missing_corner = missing_corner
        self.max_points = 4

        self.display_width = display_width
        self.display_height = display_height

        self.points = []
        self.frame = None
        self.clone = None

        self.scale_x = 1.0
        self.scale_y = 1.0

        self.points_defined = 0

    def _resize_frame(self, frame):
        h, w = frame.shape[:2]

        scale = min(
            self.display_width / w,
            self.display_height / h
        )

        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h))

        # used to convert clicked points back to original coordinates
        self.scale_x = w / new_w
        self.scale_y = h / new_h

        return resized

    def _mouse_callback(self, event, x, y, flags, param):
        
        # court edges clicked will be marked and stored
        if event == cv2.EVENT_LBUTTONDOWN:

            if len(self.points) < self.max_points:
                # convert displayed coordinates -> original frame coordinates
                original_x = int(x * self.scale_x)
                original_y = int(y * self.scale_y)

                self.points.append((original_x, original_y))

                # draw point only for visualization
                cv2.circle(self.frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(
                    self.frame,
                    str(len(self.points)),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                self.points_defined += 1

    def select(self):
        cap = cv2.VideoCapture(self.video_path)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError("Could not read video")

        resized_frame = self._resize_frame(frame)

        self.frame = resized_frame.copy()
        self.clone = resized_frame.copy()

        cv2.namedWindow("Select Keypoints", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Select Keypoints", self._mouse_callback)

        print(
            f"Click exactly {self.max_points} court points.\n"
            f"Press 'r' to reset.\n"
            f"Press 'q' to finish."
        )

        while True:
            cv2.imshow("Select Keypoints", self.frame)
            key = cv2.waitKey(1) & 0xFF

            # reset points
            if key == ord("r"):
                self.frame = self.clone.copy()
                self.points = []

            # finish selection
            if key == ord("q"):
                if len(self.points) == self.max_points:
                    break
                else:
                    print(
                        f"Need exactly {self.max_points} points, "
                        f"currently selected {len(self.points)}"
                    )

        cv2.destroyAllWindows()
        return np.array(self.points, dtype=np.float32)


"""
    Compute the homography matrix to transform video coordinates to court coordinates
"""
def get_court_homography(video_path, missing_corner):

    selector = KeypointSelector(video_path, missing_corner)
    keypoints = selector.select()

    court_pts = np.array([
        [100, 158],   # top-left
        [365, 158],   # top-right (missing)
        [100, 740],   # bottom-left
        [365, 740]   # bottom-right
    ], dtype=np.float32)

    # if the bottom left corner is 
    # missing the user must select the center point instead
    if missing_corner == 1:
        court_pts = np.array([
            [100, 158],   # top-left
            [365, 158],   # top-right
            [262, 740],   # bottom-center
            [365, 740]   # bottom-right
    ], dtype=np.float32)
        
    # if the bottom right corner is 
    # missing the user must select the center point instead
    elif missing_corner == 2:
        court_pts = np.array([
            [100, 158],   # top-left
            [365, 158],   # top-right
            [100, 740],   # bottom-left
            [262, 740],   # bottom-center
    ], dtype=np.float32)

    H, _ = cv2.findHomography(keypoints, court_pts)

    return H
