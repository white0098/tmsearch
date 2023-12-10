import os
import re
import cv2

import sys

from google.cloud import vision


def crop_padding(img, img_size_tuple, width_padding, height_padding):
    return img.crop(
        (
            width_padding,
            height_padding,
            img_size_tuple[1] + width_padding,
            img_size_tuple[0] + height_padding,
        )
    )

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
    

    img = cv2.imread(path)
    padded_img, img_size_tuple, width_padding, height_padding = preprocess_image_for_ocr(img)

    content = cv2.imencode(".jpg", padded_img)[1].tobytes()
    image = vision.Image(content=content)
    response = client.text_detection(
        image=image, image_context=vision.ImageContext(language_hints=language_hints,),
    )
    texts = response.text_annotations

    print("Texts:")
    #! duplicate area
    bbox_area = 0
    for text in texts:
        print(f'\n"{text.description}"')
        vertices = [f"({vertex.x},{vertex.y})" for vertex in text.bounding_poly.vertices]
        int_vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        bbox_area += (int_vertices[2][0] - int_vertices[0][0]) * (
            int_vertices[2][1] - int_vertices[0][1]
        )

    padded_img_area = padded_img.shape[0] * padded_img.shape[1]
    bbox_area_ratio = bbox_area / padded_img_area

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    # Draw boundin box on image
    from PIL import Image, ImageDraw

    # img = Image.open(path)
    masked_img = Image.fromarray(padded_img)
    padded_img = Image.fromarray(padded_img)
    merged_img = Image.new("RGB", padded_img.size, (255, 255, 255))
    cropped_images = []

    

    for i,text in enumerate(texts):
        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        # fill bbox with color RED
        ImageDraw.Draw(masked_img).polygon(vertices, outline="red", fill="red")
        left, top = min(vertices, key=lambda x: x[0])[0], min(vertices, key=lambda x: x[1])[1]
        right, bottom = max(vertices, key=lambda x: x[0])[0], max(vertices, key=lambda x: x[1])[1]
        
        # 이미지 잘라내기
        cropped_img = padded_img.crop((left, top, right, bottom))
        cropped_images.append((cropped_img, (left, top)))  # 잘라낸 이미지와 좌표를 저장

        if i == 0:
            design_text_img = cropped_img
            
            
            
    for cropped_img, (left, top) in cropped_images:
        # 잘라낸 이미지를 합성
        merged_img.paste(cropped_img, (left, top))
        


    # remove padding
    org_img = crop_padding(padded_img, img_size_tuple, width_padding, height_padding)
    masked_img = crop_padding(masked_img, img_size_tuple, width_padding, height_padding)
    merged_img = crop_padding(merged_img, img_size_tuple, width_padding, height_padding)

    full_text = texts[0].description

    # remove all multiple or one whitespaces and newlines
    full_text = re.sub(r"[\t\n\r\f\v]+", " ", full_text)
    full_text = re.sub(r"\s+", "", full_text)

    #! SAVE
    merged_img.save('merged_bbox.jpg')
    masked_img.save('masked_bbox.jpg')
    print(full_text)
    
    return {
        "org_img": org_img,
        "masked_img": masked_img,
        "merged_img": merged_img,
        "text": full_text,
        "bbox_area_ratio": bbox_area_ratio,
    }


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
