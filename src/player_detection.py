from ultralytics import YOLO
import cv2
import csv
import os

MAX_BOUNDING_BOXES = 2

def get_player_positions(video_path):

    # Load trained YOLO model
    model = YOLO("models/best_player.pt")

    # Open video
    cap = cv2.VideoCapture(video_path)

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output video
    out = cv2.VideoWriter(
        os.path.join("output","output_video.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    player_position = []

    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame)[0]
        boxes = results.boxes

        if len(boxes) > 0:
            # Get confidences
            confidences = boxes.conf.cpu().numpy()

            # Sort indices by confidence descending
            sorted_indices = confidences.argsort()[::-1]

            # Keep only top N boxes
            top_indices = sorted_indices[:MAX_BOUNDING_BOXES]

            for idx in top_indices:
                box = boxes[idx]

                # Coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf.cpu().numpy()[0])

                # Calculate middle point of the lower base of the bounding box
                # It is used as the current position of the player
                player_position.append([
                    int(x1) + (int(x2)-int(x1))/2,
                    int(y2)
                ])

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                label = f"{conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        # Save frame to output video
        out.write(frame)

        frame_number += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save player position to CSV
    with open(os.path.join("output","positions.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "x",
            "y"
        ])

        writer.writerows(player_position)

    return player_position
