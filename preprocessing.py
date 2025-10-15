import cv2
import numpy as np
import pytesseract
import os
from difflib import SequenceMatcher
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial


os.environ['TESSDATA_PREFIX'] = r"C:\Program Files\Tesseract-OCR\tessdata"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

dataset_path = r"dataset\test_100"
custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

def process_image(filename, success_dir, failed_dir):
    image_path = os.path.join(dataset_path, filename)
    preprocessed_img = preprocess_image(image_path)
    
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
        grayscale_img = cv2.imread(image_path, 0)
        _, lines = cv2.threshold(grayscale_img, 0, 255, cv2.THRESH_BINARY)
        preprocessed_img = grayscale_img - lines
        #grayscale_resized_img = resize_captcha(grayscale_img)
        cv2.imwrite(os.path.join(failed_dir, filename), cv2.bitwise_not(preprocessed_img))
        print(f"Incorrect: {filename} -> {extracted_text} (Expected: {expected_text})")

    return captcha_matches

# def resize_captcha(img, target_size=(380, 80)):
#     h, w = img.shape[:2]
#     target_w, target_h = target_size

#     # Maintain aspect ratio first
#     scale = target_h / h  # match height exactly
#     new_w = int(w * scale)
#     resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

#     # If the width is still too large, center-crop it
#     if new_w > target_w:
#         start_x = (new_w - target_w) // 2
#         resized = resized[:, start_x:start_x + target_w]
#     else:
#         # Otherwise, pad equally on left/right to reach target width
#         pad_left = (target_w - new_w) // 2
#         pad_right = target_w - new_w - pad_left
#         resized = cv2.copyMakeBorder(
#             resized, 0, 0, pad_left, pad_right,
#             borderType=cv2.BORDER_CONSTANT, value=0
#         )

#     return resized

def preprocess_image(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        return None
    
    # Resize all images to a fixed size (width=380, height=80)
    # target_size = (380, 80)
    # # img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    # img = resize_captcha(img, target_size)

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


    # # Apply threshold to isolate text
    # _, lines = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    # preprocessed_img = img - lines
    # preprocess_image = cv2.bitwise_not(preprocessed_img)

    # cv2.imshow("Preprocessed Image", thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cv2.bitwise_not(preprocessed_img)

def preprocess_image_with_Thresholding(image_path):
    img = cv2.imread(image_path, 0)
    if img is None:
        return None
    # Apply threshold to isolate text

    _, lines = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    preprocessed_img = img - lines

    return cv2.bitwise_not(preprocessed_img)

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

def string_similarity(ocr_text, true_text):
    true_text = true_text.lower().strip()
    ocr_text = ocr_text.lower().strip()
    ratio = SequenceMatcher(None, true_text, ocr_text).ratio()
    return ratio

def check_captcha_content(extracted_content, expected_content):
    # Convert the captcha content to lowercase for case-insensitive comparison
    extracted_content = extracted_content.lower()
    expected_content = expected_content.replace(" ", "")

    # Check similarity using Levenshtein distance
    similarity = levenshtein_similarity(extracted_content, expected_content)
    # similarity = string_similarity(extracted_content, expected_content)
    return similarity >= 0.5  # Threshold for considering a match

if __name__ == "__main__":
    success_dir = os.path.join(dataset_path, "processed_success")
    failed_dir = os.path.join(dataset_path, "processed_failed")
    # Gather all .png files
    png_files = [f for f in os.listdir(dataset_path) if f.endswith(".png")]
    if not os.path.exists(success_dir):
        os.makedirs(success_dir)
    if not os.path.exists(failed_dir):
        os.makedirs(failed_dir)
    correct_predictions = 0
    total_images = len(png_files)

    # Wrap the function
    wrapped_fn = partial(process_image, success_dir=success_dir, failed_dir=failed_dir)
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(wrapped_fn, png_files), total=len(png_files)))

    correct_predictions = sum(results)
    print(f"Total Images: {total_images}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {correct_predictions}/{total_images} = {correct_predictions / total_images:.2%}")

# accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
# print(f"Total Images: {total_images}")
# print(f"Correct Predictions: {correct_predictions}")
# print(f"Accuracy: {accuracy:.2f}%")
