import math
import os
import sys
import cv2
import scipy
import skimage
import tifffile
import toml
import numpy as np
import filedialpy


# config path setting
app_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(app_dir, "config.toml")

# create config file if not exists
if not os.path.exists(config_path):
    with open(config_path, "w"):
        pass

# load config
config = toml.load(config_path)


# try to get last_dir and fallback to home dir
def get_last_dir():
    try:
        last_dir = config["last_dir"]
    except Exception as e:
        last_dir = os.path.expanduser("~")

    return last_dir


# dialog to select input folder. exit if nothing is selected
input_dir = filedialpy.openDir(title="Choose Input Folder", initial_dir=get_last_dir())
if not input_dir or input_dir == "":
    print("no folder selected. exiting.")
    sys.exit(1)

group_name = os.path.basename(input_dir)


# update last_dir
config["last_dir"] = input_dir
with open(config_path, "w") as f:
    toml.dump(config, f)


# get all tiff file in folder
tiff_files = []
for root, dirs, files in os.walk(input_dir):
    # Skip folders with 'crop' in their name
    if "crop" in os.path.basename(root).lower():
        continue
    for f in files:
        if f.lower().endswith((".tiff", ".tif", ".TIFF", ".TIF")):
            tiff_files.append(os.path.relpath(os.path.join(root, f), input_dir))

# gather all tifffiles,
# identify the largest object on image,
# skeletonize the object,
# find the two tips (the farest two points on the skeleton)
# find the head (the greener tip)
# rotate the image to make the two tips aligned with vertical axis
# crop out the largest object (size: 1800x600)
cropped_images = []
for tiff_file in tiff_files:
    # read data
    tiff_path = os.path.join(input_dir, tiff_file)
    print(f"processing {tiff_path} ...")
    image = tifffile.imread(tiff_path)

    # get image mask for the largest object
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_bin = cv2.threshold(image_gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[
        1
    ]
    image_bin = scipy.ndimage.binary_fill_holes(image_bin)
    labels = skimage.measure.label(image_bin)
    rp = skimage.measure.regionprops(labels)
    largest_label = -1
    max_area = 0
    for prop in rp:
        if prop.area > max_area:
            max_area = prop.area
            largest_label = prop.label

    if largest_label == -1:
        print(f"not object detected in file {tiff_path}")
        continue
    image_mask = (labels == largest_label).astype(np.uint8) * 255
    # # make the boarder of mask more smooth
    # image_mask = scipy.ndimage.binary_erosion(image_mask, iterations=3)

    # Erode the mask to shrink the object
    eroded_mask = cv2.erode(image_mask, np.ones((3, 3), np.uint8), iterations=1)

    image_border = image_mask - eroded_mask

    # find the two tips (the farest two points on the border)
    # Get coordinates of border pixels
    mask_points = np.column_stack(np.where(image_border > 0))

    # Compute pairwise distances
    if len(mask_points) < 2:
        print(f"not enough border points in file {tiff_path}")
        continue

    dists = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(mask_points))
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    tip1 = tuple(mask_points[i])
    tip2 = tuple(mask_points[j])

    # identify the head, the greener tip
    # Extract green channel
    green_channel = image[:, :, 1]

    # Define a small region around each tip (e.g., 10x10 square)
    # Compute 0.2 and 0.8 points along the line between tip1 and tip2
    tip1 = np.array(tip1)
    tip2 = np.array(tip2)
    point_02 = tip1 + 0.2 * (tip2 - tip1)
    point_08 = tip1 + 0.8 * (tip2 - tip1)

    # Round to nearest integer pixel coordinates
    point_02 = np.round(point_02).astype(int)
    point_08 = np.round(point_08).astype(int)

    # Get green intensity at those points
    def get_green_at_point(point, green_channel, region_size=10):
        x, y = point
        half = region_size // 2
        x_min = max(0, x - half)
        x_max = min(green_channel.shape[0], x + half + 1)
        y_min = max(0, y - half)
        y_max = min(green_channel.shape[1], y + half + 1)
        region = green_channel[x_min:x_max, y_min:y_max]
        if region.size == 0:
            return 0
        return np.mean(region)

    green_02 = get_green_at_point(point_02, green_channel)
    green_08 = get_green_at_point(point_08, green_channel)

    # Assign head and tail based on green intensity
    if green_02 > green_08:
        head_tip = tuple(tip1)
        tail_tip = tuple(tip2)
    else:
        head_tip = tuple(tip2)
        tail_tip = tuple(tip1)

    # put the raw image on a 1800x600x3 image
    # rotate to make tail on bottom, head on top
    # make pixel size unchange
    # make the center of the two tips at the center of cropped image
    # always use the information from raw image
    # unless the transformed pixel is outside of the original image
    # Get raw image dimensions
    raw_height, raw_width, _ = image.shape

    # Calculate the angle of rotation based on bottom and top points
    dx = head_tip[1] - tail_tip[1]
    dy = head_tip[0] - tail_tip[0]
    angle_rad = math.atan2(dy, dx)  # Angle in radians
    angle_deg = math.degrees(angle_rad)  # Convert to degrees
    center_x = int((head_tip[1] + tail_tip[1]) / 2)
    center_y = int((head_tip[0] + tail_tip[0]) / 2)

    # Target angle for vertical up (head on top, negative y direction)
    target_angle_deg = -90.0
    rotation_deg = target_angle_deg - angle_deg

    # Center point
    center = np.array([center_y, center_x])  # [y, x]

    # Function to compute matrix and offset
    def compute_transform(rotation_deg):
        angle_rad = math.radians(rotation_deg)
        cos = math.cos(angle_rad)
        sin = math.sin(angle_rad)
        rotation_matrix = np.array([[cos, -sin], [sin, cos]])
        offset = center - rotation_matrix @ center
        inv_matrix = np.array([[cos, sin], [-sin, cos]])  # Inverse for mapping points
        return rotation_matrix, inv_matrix, offset

    rotation_matrix, inv_matrix, offset = compute_transform(rotation_deg)

    # Map the top and bottom points to check orientation
    top_pos = np.array(head_tip)
    bottom_pos = np.array(tail_tip)
    mapped_top = inv_matrix @ (top_pos - offset)
    mapped_bottom = inv_matrix @ (bottom_pos - offset)

    # If head (top) is not above tail (bottom), flip by 180 degrees
    if mapped_top[0] > mapped_bottom[0]:
        rotation_deg += 180.0
        rotation_matrix, inv_matrix, offset = compute_transform(rotation_deg)

    # Compute new bounds for full rotated image without clipping
    corners = np.array(
        [
            [0, 0],
            [0, raw_width - 1],
            [raw_height - 1, 0],
            [raw_height - 1, raw_width - 1],
        ]
    )  # [y, x]

    mapped_corners = np.array([inv_matrix @ (corner - offset) for corner in corners])

    min_y = np.floor(mapped_corners[:, 0].min())
    min_x = np.floor(mapped_corners[:, 1].min())
    max_y = np.ceil(mapped_corners[:, 0].max())
    max_x = np.ceil(mapped_corners[:, 1].max())

    new_height = int(max_y - min_y + 1)
    new_width = int(max_x - min_x + 1)

    trans = np.array([min_y, min_x])
    offset_new = offset + rotation_matrix @ trans

    # Apply affine transform to each channel
    rotated_array = np.zeros((new_height, new_width, 3), dtype=np.float64)
    for c in range(3):
        rotated_array[:, :, c] = scipy.ndimage.affine_transform(
            image[:, :, c],
            rotation_matrix,
            offset_new,
            output_shape=(new_height, new_width),
            order=1,  # Bilinear interpolation
            mode="constant",
            cval=0.0,
        )
    rotated_array = np.clip(rotated_array, 0, 255).astype(np.uint8)

    # New center position in rotated image
    new_center_y = center_y - min_y
    new_center_x = center_x - min_x

    # Crop to 1800 height x 600 width around the new center
    half_height = 900  # 1800 / 2
    half_width = 300  # 600 / 2

    top_crop = max(0, int(new_center_y - half_height))
    bottom_crop = min(new_height, int(new_center_y + half_height))
    left_crop = max(0, int(new_center_x - half_width))
    right_crop = min(new_width, int(new_center_x + half_width))

    crop_height = bottom_crop - top_crop
    crop_width = right_crop - left_crop
    cropped_array = rotated_array[top_crop:bottom_crop, left_crop:right_crop]

    # Pad to exactly 1800x600 if smaller
    output_array = np.zeros((1800, 600, 3), dtype=np.uint8)
    pad_top = max(0, (1800 - crop_height) // 2)
    pad_left = max(0, (600 - crop_width) // 2)
    output_array[pad_top : pad_top + crop_height, pad_left : pad_left + crop_width] = (
        cropped_array
    )

    cropped_images.append((tiff_path, output_array))

if len(cropped_images) == 0:
    print("no sucessful processed image. exiting")
    sys.exit(2)

# save cropped_images to a subfolder named "crop"
crop_dir = os.path.join(input_dir, "crop")
os.makedirs(crop_dir, exist_ok=True)
for i, (tiff_path, cropped_image) in enumerate(cropped_images):
    base = os.path.splitext(os.path.basename(tiff_path))[0]
    save_path = os.path.join(crop_dir, f"{base}_cropped.tif")
    print(f"saving {save_path} ...")
    tifffile.imwrite(save_path, cropped_image)

# save a montage in 10 column format
# width 600 * min(10, n_image) = 1000
# height 1800 * int(n_image / 10)
if len(cropped_images) > 0:
    n_cols = min(10, len(cropped_images))
    n_rows = int(np.ceil(len(cropped_images) / n_cols))
    montage_array = np.zeros((1800 * n_rows, 600 * n_cols, 3), dtype=np.uint8)

    for i, (tiff_path, cropped_image) in enumerate(cropped_images):
        row = i // n_cols
        col = i % n_cols
        montage_array[row * 1800 : (row + 1) * 1800, col * 600 : (col + 1) * 600] = (
            cropped_image
        )

    montage_path = os.path.join(input_dir, f"montage_{group_name}_{len(cropped_images)}.png")
    print(f"saving montage {montage_path} ...")
    cv2.imwrite(montage_path, cv2.cvtColor(montage_array, cv2.COLOR_RGB2BGR))
