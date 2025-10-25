# Final preprocessing to resize all images to a consistent width and height
# 780x80 or shape (80, 780)
# Number of one hot vectors required = 8

import cv2
import os
import pytesseract

# Find the longest image width
def get_longest_images(folder):
  image_files = [f for f in os.listdir(folder) if f.lower().endswith((".png"))]
  longest = 0
  for i in range(len(image_files)):
    img = cv2.imread(folder + "/" + image_files[i], 0)
    longest = max(longest, img.shape[1])

  return longest

# Get longest label length to find number of one-hot vectors for final CAPTCHA string
def get_longest_label_length(folder):
    image_files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    max_len = 0
    longest_label = ""

    for fname in image_files:
        # Split at '-0.' to isolate the captcha text
        label = fname.split('-')[0]
        if len(label) > max_len:
            max_len = len(label)
            longest_label = label

    print(f"Longest label: {longest_label} (length {max_len})")
    return max_len

def pad_image(img, target_width=780, target_height=80):
    h, w = img.shape[:2]
    if w >= target_width:
        return cv2.resize(img, (target_width, target_height))
    pad_right = target_width - w
    img_resized = cv2.resize(img, (w, target_height))
    padded = cv2.copyMakeBorder(
        img_resized, 0, 0, 0, pad_right,
        cv2.BORDER_CONSTANT, value=[255,255,255]  # white background
    )
    return padded

def process_label(img, label):
    ocr_text = pytesseract.image_to_string(img, config='--psm 7').strip()
    print(ocr_text)
    # only compare if lengths match
    if len(ocr_text) == len(label):
        merged = []
        for lch, och in zip(label, ocr_text):
            # same letter ignoring case?
            if lch.lower() == och.lower():
                # if OCR saw uppercase, keep uppercase
                merged.append(och if och.isupper() else lch)
            else:
                # mismatch → keep original label char
                merged.append(lch)
        return "".join(merged)
    return label

def pad_images(folder, output, target_width=780, target_height=80):
  os.makedirs(output, exist_ok=True)

  image_files = [f for f in os.listdir(folder) if f.lower().endswith((".png"))]

  for i in range(len(image_files)):
    img = cv2.imread(folder + "/" + image_files[i], 0)
    padded_image = pad_image(img, target_width, target_height)
    label = image_files[i].split("-")[0]
    label = process_label(img, label)

    print(label)

    cv2.imwrite(output + "/" + label, padded_image)

  
if __name__ == "__main__":
  paths = ["dataset/test/processed_success", "dataset/train/processed_success"]
  # print([get_longest_images(path) for path in paths])

  # ^ Longest image is 780x80  
  pad_images(paths[0], "test")  
  pad_images(paths[1], "train")

  # train_max = get_longest_label_length("dataset/train/processed_success")
  # test_max = get_longest_label_length("dataset/test/processed_success")
  # print(f"Overall max length: {max(train_max, test_max)}")