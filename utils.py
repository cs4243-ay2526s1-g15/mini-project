import cv2
import numpy as np
from difflib import SequenceMatcher

# Resize function can be used to have consistent captcha sizes
def resize_captcha(img, target_size):
    h, w = img.shape[:2]
    target_w, target_h = target_size

    # Maintain aspect ratio first
    scale = target_h / h  # match height exactly
    new_w = int(w * scale)
    resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    # If the width is still too large, center-crop it
    if new_w > target_w:
        start_x = (new_w - target_w) // 2
        resized = resized[:, start_x:start_x + target_w]
    else:
        # Otherwise, pad equally on left/right to reach target width
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        resized = cv2.copyMakeBorder(
            resized, 0, 0, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=0
        )

    return resized

# String similarity function
def string_similarity(ocr_text, true_text):
    true_text = true_text.lower().strip()
    ocr_text = ocr_text.lower().strip()
    ratio = SequenceMatcher(None, true_text, ocr_text).ratio()
    return ratio