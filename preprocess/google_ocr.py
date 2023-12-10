import os
import re
import cv2

import sys

from google.cloud import vision


def preprocess_image_for_ocr(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # padding 20% of image size
    img_size_tuple = img.shape
    width_padding = int(img_size_tuple[1] * 0.2)
    height_padding = int(img_size_tuple[0] * 0.2)
    img = cv2.copyMakeBorder(
        img,
        height_padding,
        height_padding,
        width_padding,
        width_padding,
        cv2.BORDER_CONSTANT,
    )
    return img, img_size_tuple, width_padding, height_padding


def detect_document(path):
    """Detects document features in an image."""

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            print(f"\nBlock confidence: {block.confidence}\n")

            for paragraph in block.paragraphs:
                print("Paragraph confidence: {}".format(paragraph.confidence))

                for word in paragraph.words:
                    word_text = "".join([symbol.text for symbol in word.symbols])
                    print(
                        "Word text: {} (confidence: {})".format(
                            word_text, word.confidence
                        )
                    )

                    for symbol in word.symbols:
                        print(
                            "\tSymbol: {} (confidence: {})".format(
                                symbol.text, symbol.confidence
                            )
                        )

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )


def detect_text(path, language_hints=None):
    """Detects text in the file."""

    from google.cloud import vision


    client = vision.ImageAnnotatorClient()

    language_hints = ["ko", "en"]

    # add padding, resize to 224x224
    import cv2

    img = cv2.imread(path)
    img, img_size_tuple, width_padding, height_padding = preprocess_image_for_ocr(img)

    content = cv2.imencode(".jpg", img)[1].tobytes()
    image = vision.Image(content=content)
    response = client.text_detection(
        image=image, image_context=vision.ImageContext(language_hints=language_hints,),
    )
    texts = response.text_annotations
    print("Texts:")

    for text in texts:
        print(f'\n"{text.description}"')

        vertices = [f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices]

        print("bounds: {}".format(",".join(vertices)))

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    # Draw boundin box on image
    from PIL import Image, ImageDraw

    # img = Image.open(path)
    masked_img = Image.fromarray(img)
    img = Image.fromarray(img)

    for text in texts:
        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        # fill bbox with color RED
        ImageDraw.Draw(masked_img).polygon(vertices, outline="red", fill="red")
    # remove padding
    org_img = img.crop(
        (
            width_padding,
            height_padding,
            img_size_tuple[1] + width_padding,
            img_size_tuple[0] + height_padding,
        )
    )
    masked_img = masked_img.crop(
        (
            width_padding,
            height_padding,
            img_size_tuple[1] + width_padding,
            img_size_tuple[0] + height_padding,
        )
    )
    full_text = texts[0].description

    # remove all multiple or one whitespaces and newlines

    full_text = re.sub(r"[\t\n\r\f\v]+", " ", full_text)
    full_text = re.sub(r"\s+", "", full_text)

    #! SAVE
    masked_img.save('masked_bbox.jpg')
    print(full_text)
    
    return org_img, masked_img, full_text


def detect_logos(path):

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.logo_detection(image=image)
    logos = response.logo_annotations
    print("Logos:")

    for logo in logos:
        print(logo.description)

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )


if __name__ == "__main__":
    detect_text(sys.argv[1])
    # detect_logos(sys.argv[1])
    # detect_document(sys.argv[1])
