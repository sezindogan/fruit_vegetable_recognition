import os
import cv2
import logging
import numpy as np
import pandas as pd 
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    filename="log.txt",
    filemode="w"
)

logger = logging.getLogger(__name__)

folder_path = "data/raw"
output_csv = 'data/processed/image_features.csv'

def remove_background(img, sat_thresh=30, blur=5):
    """
    Removes white background using HSV saturation thresholding.
    """

    # get saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)

    # threshold and blur
    _, mask = cv2.threshold(S, sat_thresh, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(mask, blur)

    # keep only the largest object (the fruit)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8) * 255

    # apply mask
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    return mask, masked_img

def plot_masked(img, mask, masked):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(img); plt.title("Original"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(mask, cmap="gray"); plt.title("Mask"); plt.axis("off")
    plt.subplot(1,3,3); plt.imshow(masked); plt.title("Masked Image"); plt.axis("off")
    plt.show()

def lbp_features(gray_masked, P=8, R=1, bins=8):
    """
    Computes coarse LBP histogram (bins=8) for texture feature.
    """
    lbp = local_binary_pattern(gray_masked, P=P, R=R, method='uniform')
    values = lbp[gray_masked > 0] # ignore background
    if len(values) == 0:
        print("no object found. unsuccessful")
        return np.zeros(bins)
    hist, _ = np.histogram(values, bins=bins, range=(0, P+2), density=True)
    return hist


def extract_features(img, mask):
    """
    Extracts the following features:
        - Lab means,
        - HSV means,
        - RGB means,
        - LBP histogram,
        - aspect_ratio, circularity
    """
    if not (mask > 0).any():
        return None

    mask_bool = mask > 0

    # color features
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    L, A, B = cv2.split(lab)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)

    R, G, B = cv2.split(img)

    def masked_mean(x):
        vals = x[mask_bool]
        return float(np.mean(vals)) if len(vals) else 0.0

    features = []

    # Lab
    features.append(masked_mean(L))
    features.append(masked_mean(A))
    features.append(masked_mean(B))

    # HSV
    features.append(masked_mean(H))
    features.append(masked_mean(S))
    features.append(masked_mean(V))

    # RGB
    features.append(masked_mean(R))
    features.append(masked_mean(G))
    features.append(masked_mean(B))

    # texture features
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp_hist = lbp_features(gray, bins=8)
    features.extend(lbp_hist.tolist())

    # shape features
    ys, xs = np.where(mask_bool)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    aspect_ratio = w / h

    area = np.sum(mask_bool)
    edges = cv2.Canny(mask.astype(np.uint8) * 255, 50, 150)
    perimeter = np.sum(edges > 0)
    circularity = (4 * np.pi * area) / (perimeter**2 + 1e-6)

    features.extend([aspect_ratio, circularity])

    return np.array(features, dtype=np.float32)


def main():
    rows = []

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith((".jpg")):
            continue

        # open image
        path = os.path.join(folder_path, file_name)
        img_bgr = cv2.imread(path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # remove background
        mask, masked_img = remove_background(img)

        # extract features
        features = extract_features(img, mask)
        if features is None:
            logger.info(f"Skipping image {file_name} with empty mask!")

        # create row
        row = {"filename": file_name}
        for i, f in enumerate(features):
            row[f"f{i}"] = float(f)

        rows.append(row)
        logger.info(f"Image {file_name} is successfully proccessed.")

    # convert to df and save
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    logger.info("Feature DataFrame saved:", output_csv)


if __name__ == "__main__":
    main()