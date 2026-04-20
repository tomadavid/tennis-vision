import os
from court_coordinates import get_court_homography
from player_detection import get_player_positions
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import sys


if __name__ == "__main__":

    # Expected usage:
    # python script.py <video_path> <missing_corner> [positions.csv]

    if len(sys.argv) < 3:
        print("Usage: python script.py <video_path> <missing_corner> [positions.csv]")
        print("missing_corner must be: 0, 1, or 2")
        sys.exit(1)

    video_path = sys.argv[1]
    missing_corner = sys.argv[2]

    # Optional third argument
    positions_path = sys.argv[3] if len(sys.argv) > 3 else None

    # Validate missing_corner
    if missing_corner not in ["0", "1", "2"]:
        print("Invalid missing corner argument. Use '0', '1', or '2'.")
        sys.exit(1)

    # Validate optional CSV path
    if positions_path is not None and not positions_path.endswith(".csv"):
        print("Invalid positions file. Please provide a CSV file.")
        sys.exit(1)

    if not os.path.exists("../output"):
        os.makedirs("output")

    # Load positions
    if positions_path is not None:
        df = pd.read_csv(positions_path)

        # Validate required columns
        if not {"x", "y"}.issubset(df.columns):
            print("CSV file must contain 'x' and 'y' columns.")
            sys.exit(1)

        positions = df[["x", "y"]].values.tolist()

    else:
        # Run detection if CSV not provided
        positions = get_player_positions(video_path)

    H = get_court_homography(video_path, missing_corner)
    
    
    player_court_coordinates = []

    # convert video coordinates to court coordinates
    for pos in positions:
        point = np.array([[[pos[0], pos[1]]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, H)

        X, Y = mapped[0][0]
        player_court_coordinates.append((X, Y))

    # plot tennis court and player positions
    court_img = plt.imread("../tennis_court.webp")

    # plot bounds (same as court image dimensions)
    x_min, x_max = 0, 467
    y_min, y_max = 0, 900

    x_coords = np.array([x for x, y in player_court_coordinates])
    y_coords = np.array([y for x, y in player_court_coordinates])

    # filter points inside court (handle outliers)
    mask = (
        (x_coords >= x_min) & (x_coords <= x_max) &
        (y_coords >= y_min) & (y_coords <= y_max)
    )

    x_coords = x_coords[mask]
    y_coords = y_coords[mask]

    plt.figure(figsize=(8, 12))

    # Display image with the court image coordinate system
    plt.imshow(
        court_img,
        extent=[0, 467, 900, 0]
    )

    # Smooth density heatmap (instead of points)
    plt.hexbin(
        x_coords,
        y_coords,
        gridsize=40,
        mincnt=1,
        alpha=0.6
    )

    plt.colorbar(label="Player Density")

    plt.title("Player Movement Heatmap")
    plt.show()