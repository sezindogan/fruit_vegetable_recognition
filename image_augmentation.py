import os
import cv2
import albumentations as A

input_dir = "data/raw"
output_dir = "data/processed"

#photometric
photometric = A.OneOf([
    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.1,
        p=0.5
    ),
    A.GaussNoise(
        var_limit=(0.05, 0.1),
        mean=0.1,
        per_channel=True,
        p=0.5
    )
], p=1)


# geometric
geometric = A.Compose([
    A.Rotate(
        limit=15,
        border_mode=cv2.BORDER_REPLICATE,
        p=0.7
    ),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=0,
        border_mode=cv2.BORDER_REPLICATE,
        p=0.6
    ),
    A.HorizontalFlip(p=0.6)
])

AUG = A.Compose([
    geometric,
    photometric
])

# augment single image
def augment_image(img, num_aug=3):
    aug_images = []
    for _ in range(num_aug):
        augmented = AUG(image=img)
        aug_images.append(augmented["image"])
    return aug_images

# augment dataset
def augment_dataset(
    input_folder,
    output_folder,
    num_aug=3,
    exts=(".jpg", ".jpeg", ".png")
):
    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(exts):
            continue

        path = os.path.join(input_folder, fname)
        img = cv2.imread(path)

        if img is None:
            print("Could not read:", fname)
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # augment
        augmented_imgs = augment_image(img, num_aug=num_aug)

        # save original
        base = os.path.splitext(fname)[0]
        """cv2.imwrite(
            os.path.join(output_folder, f"{base}_orig.jpg"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        )"""

        # save augmentations
        for i, aug in enumerate(augmented_imgs):
            cv2.imwrite(
                os.path.join(output_folder, f"{base}_aug{i+1}.jpg"),
                cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)
            )

        print("Processed:", fname)

    print("Done.")


if __name__ == "__main__":
    augment_dataset(input_dir, output_dir)
