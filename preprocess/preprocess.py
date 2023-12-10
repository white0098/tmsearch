from preprocess.google_ocr import detect_text


def preprocess_image(image_path):
    return detect_text(image_path)
