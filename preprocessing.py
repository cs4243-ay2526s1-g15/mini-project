import cv2
import numpy as np
import pytesseract
import os
import shutil
from difflib import SequenceMatcher
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial

# PyTesseract configuration
os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdata"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

# Path to the dataset
dataset_path = r"dataset\train"

# Levenshtein similarity function for comparing captcha text with filename
def levenshtein_similarity(s1, s2):
    if len(s1) < len(s2):
        s1, s2 = s2, s1  # ensure s1 is longer

    # If one string is empty
    if len(s2) == 0:
        return 0 if len(s1) > 0 else 1

    # Initialize distance matrix
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        current_row = [i]
        for j, c2 in enumerate(s2, 1):
            insertions = previous_row[j] + 1
            deletions = current_row[j-1] + 1
            substitutions = previous_row[j-1] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    distance = previous_row[-1]
    max_len = max(len(s1), len(s2))
    similarity = (max_len - distance) / max_len
    return similarity

# Remove lines 
def remove_black_lines_and_inpaint(image_path, inpaint_radius=1):
    """
    img: input color image (BGR)
    inpaint_radius: radius for inpainting

    Returns: color image with black lines removed and intensity interpolated
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    # Step 1: Create mask for black lines
    # A pixel is considered black if all channels are very low
    black_mask = cv2.inRange(img, (0, 0, 0), (1, 1, 1))  # 0-10 intensity = black lines

    # Step 2: Inpaint the image
    # cv2.inpaint expects a single-channel mask (255 = pixels to restore)
    inpainted = cv2.inpaint(img, black_mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)

    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)

def preprocess_image(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        return None

    # Denoise slightly with a median filter (good for line noise and dots)
    img = cv2.medianBlur(img, 3)

    # Apply contrast enhancement using histogram equalization
    img = cv2.equalizeHist(img)

    # Adaptive thresholding handles uneven lighting/backgrounds
    preprocessed_img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # text = white on black background
        11, 5
    )

    return cv2.bitwise_not(preprocessed_img)

# Third layer preprocessing function with morphological operations
def preprocess_image_third_layer(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        return None

    # Step 1: Denoise with bilateral filter (preserve edges better than median)
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 2: Morphological opening to remove thin lines
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

    # Step 3: Adaptive threshold again, but inverted (for better Tesseract detection)
    thresh = cv2.adaptiveThreshold(
        opened, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3
    )

    # Step 4: Remove small noise blobs (using contour filtering)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(thresh)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20:  # keep only reasonable character-sized blobs
            cv2.drawContours(mask, [cnt], -1, 255, -1)
    cleaned = cv2.bitwise_and(thresh, mask)

    # Step 5: Optional sharpening to make edges crisp
    kernel_sharp = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
    sharpened = cv2.filter2D(cleaned, -1, kernel_sharp)

    return cv2.bitwise_not(sharpened)

# Function to check if the extracted captcha content matches the expected content using levenshtein similarity
def check_captcha_content(extracted_content, expected_content):
    # Convert the captcha content to lowercase for case-insensitive comparison
    extracted_content = extracted_content.lower()
    expected_content = expected_content.replace(" ", "")

    # Check similarity using Levenshtein distance
    similarity = levenshtein_similarity(extracted_content, expected_content)

    return similarity >= 0.5  # Threshold for considering a match

# First layer processing function to check captcha content after removing lines
def process_image_first_layer(filename, success_dir, failed_dir):
    filename = os.path.basename(filename)
    image_path = os.path.join(dataset_path, filename)
    preprocessed_img = remove_black_lines_and_inpaint(image_path)
    
    if preprocessed_img is None:
        print(f"Failed to load image: {filename}")
        return False

    extracted_text = pytesseract.image_to_string(preprocessed_img, config=custom_config).strip()
    expected_text = filename.split('-0.')[0]

    captcha_matches = check_captcha_content(extracted_text, expected_text)

    if captcha_matches:
        cv2.imwrite(os.path.join(success_dir, filename), preprocessed_img)
        print(f"Correct: {filename} -> {extracted_text}")
    else:
        cv2.imwrite(os.path.join(failed_dir, filename), preprocessed_img)
        print(f"Incorrect: {filename} -> {extracted_text} (Expected: {expected_text})")

    return captcha_matches

# Second layer processing function to check captcha content after additional preprocessing
def process_image_second_layer(filename, success_dir, failed_dir):
    filename = os.path.basename(filename)
    src_image_path = os.path.join(failed_dir, filename)
    dst_image_path = os.path.join(success_dir, filename)

    preprocessed_img = preprocess_image(src_image_path)

    if preprocessed_img is None:
        print(f"Failed to load image: {filename}")
        return False
    
    extracted_text = pytesseract.image_to_string(preprocessed_img, config=custom_config).strip()
    expected_text = filename.split('-0.')[0]

    captcha_matches = check_captcha_content(extracted_text, expected_text)

    if captcha_matches:
        shutil.move(src_image_path, dst_image_path)
        print(f"Correct on 2nd layer: {filename} -> {extracted_text}")
    else:
        print(f"Still Incorrect: {filename} -> {extracted_text} (Expected: {expected_text})")

    return captcha_matches

# Third layer processing function to check captcha content after advanced preprocessing
def process_image_third_layer(filename, success_dir, failed_dir):
    filename = os.path.basename(filename)
    src_image_path = os.path.join(failed_dir, filename)
    dst_image_path = os.path.join(success_dir, filename)

    preprocessed_img = preprocess_image_third_layer(src_image_path)
    if preprocessed_img is None:
        print(f"Failed to load image: {filename}")
        return False

    extracted_text = pytesseract.image_to_string(preprocessed_img, config=custom_config).strip()
    expected_text = filename.split('-0.')[0]

    captcha_matches = check_captcha_content(extracted_text, expected_text)

    if captcha_matches:
        shutil.move(src_image_path, dst_image_path)
        print(f"Correct on 3rd layer: {filename} -> {extracted_text}")
    else:
        print(f"Still Incorrect after 3 layers: {filename} -> {extracted_text} (Expected: {expected_text})")

    return captcha_matches

# Main function to process all images in the dataset
if __name__ == "__main__":
    success_dir = os.path.join(dataset_path, "processed_success")
    failed_dir = os.path.join(dataset_path, "processed_failed")

    # Create directories if they don't exist
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)

    # Gather all .png files
    png_files = [f for f in os.listdir(dataset_path) if f.endswith(".png")]
    total_images = len(png_files)
    correct_predictions = 0

    # First layer processing for images
    wrapped_fn_1 = partial(process_image_first_layer, success_dir=success_dir, failed_dir=failed_dir)
    with ProcessPoolExecutor() as executor:
        results_first_layer = list(tqdm(executor.map(wrapped_fn_1, png_files), total=total_images))

    correct_predictions = sum(results_first_layer)
    print(f"Total Images: {total_images}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {correct_predictions}/{total_images} = {correct_predictions / total_images:.2%}")

    failed_png_files = [f for f in os.listdir(failed_dir) if f.endswith(".png")]
    total_failed_images = len(failed_png_files)
    correct_predictions_second_layer = 0

    # Second layer processing for failed images
    wrapped_fn_2 = partial(process_image_second_layer, success_dir=success_dir, failed_dir=failed_dir)
    with ProcessPoolExecutor() as executor:
        results_second_layer = list(tqdm(executor.map(wrapped_fn_2, failed_png_files), total=total_failed_images))

    correct_predictions_second_layer = sum(results_second_layer)
    print(f"Total Failed Images: {total_failed_images}")
    print(f"Correct Predictions in 2nd Layer: {correct_predictions_second_layer}")
    print(f"Total Correct Predictions: {correct_predictions + correct_predictions_second_layer}")

    failed_png_files = [f for f in os.listdir(failed_dir) if f.endswith(".png")]
    total_failed_images = len(failed_png_files)
    correct_predictions_third_layer = 0

    wrapped_fn_3 = partial(process_image_third_layer, success_dir=success_dir, failed_dir=failed_dir)
    with ProcessPoolExecutor() as executor:
        results_third_layer = list(tqdm(executor.map(wrapped_fn_3, failed_png_files), total=total_failed_images))

    correct_predictions_third_layer = sum(results_third_layer)
    print(f"Correct Predictions in 3rd Layer: {correct_predictions_third_layer}")
    print(f"Total Correct Predictions (All Layers): {correct_predictions + correct_predictions_second_layer + correct_predictions_third_layer}")
    print(f"Correct Predictions in 2nd Layer: {correct_predictions_second_layer}")
    print(f"Correct Predictions: {correct_predictions}")


