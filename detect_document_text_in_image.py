from PIL import Image, ImageDraw
from enum import Enum
from google.cloud import vision
import io
import cv2
import numpy as np


class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5


def draw_boxes(image, bounds, color):
    """Draw a border around the image using the hints in the vector list."""
    draw = ImageDraw.Draw(image)

    for bound in bounds:
        draw.polygon([
            bound.vertices[0].x, bound.vertices[0].y,
            bound.vertices[1].x, bound.vertices[1].y,
            bound.vertices[2].x, bound.vertices[2].y,
            bound.vertices[3].x, bound.vertices[3].y], None, color)
    return image


def get_document_bounds(image_file, feature):
    """Returns document bounds given an image."""
    client = vision.ImageAnnotatorClient()

    bounds = []

    with io.open(image_file, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)
    document = response.full_text_annotation

    # Collect specified feature bounds by enumerating all document features
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        if (feature == FeatureType.SYMBOL):
                            bounds.append(symbol.bounding_box)

                    if (feature == FeatureType.WORD):
                        bounds.append(word.bounding_box)

                if (feature == FeatureType.PARA):
                    bounds.append(paragraph.bounding_box)

            if (feature == FeatureType.BLOCK):
                bounds.append(block.bounding_box)

    # The list `bounds` contains the coordinates of the bounding boxes.
    return bounds


def render_doc_text(filein, fileout):
    image = Image.open(filein)
    # bounds = get_document_bounds(filein, FeatureType.BLOCK)
    # draw_boxes(image, bounds, 'red')
    bounds = get_document_bounds(filein, FeatureType.PARA)
    draw_boxes(image, bounds, 'red')
    # bounds = get_document_bounds(filein, FeatureType.WORD)
    # draw_boxes(image, bounds, 'blue')

    if fileout != 0:
        image.save(fileout)
    else:
        image.show()


def find_lines(image_path):
    client = vision.ImageAnnotatorClient()

    lines = []

    with io.open(image_path, 'rb') as image_path:
        content = image_path.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)
    document = response.full_text_annotation
    for page in document.pages:
        for block in page.blocks:
            for paragraph_index in range(len(block.paragraphs)):
                paragraph = block.paragraphs[paragraph_index]
                for word_index in range(len(paragraph.words)):
                    word = paragraph.words[word_index]
                    for symbol_index in range(len(word.symbols)):
                        symbol = word.symbols[symbol_index]
                        if symbol.property.detected_break:
                            if str(symbol.property.detected_break.type_) != "BreakType.SPACE":
                                next_bounding_box = find_next_line_break(block, paragraph_index, word_index)
                                if next_bounding_box is not None:
                                    if not are_line_breaks_too_close(symbol.bounding_box, next_bounding_box):
                                        # draw across entire line, so reduce x to start of line
                                        symbol.bounding_box.vertices[0].x = 10
                                        symbol.bounding_box.vertices[3].x = 10
                                        lines.append(symbol.bounding_box)
                                else:
                                    # draw across entire line, so reduce x to start of line
                                    symbol.bounding_box.vertices[0].x = 10
                                    symbol.bounding_box.vertices[3].x = 10
                                    lines.append(symbol.bounding_box)
    return lines


def perform_finding_lines(image_path):
    lines = find_lines(image_path)
    img = Image.open(image_path)
    new_img = np.array(draw_boxes(img, lines, 'red'))
    cv2.imwrite('static/uploaded_img_res.jpg', new_img)
    return lines

def find_next_line_break(block, paragraph_index, word_index):
    """

    :param block:
    :param paragraph_index:
    :param word_index:
    :return:
    """

    # search in current paragraph first
    curr_paragraph = block.paragraphs[paragraph_index]
    for i in range(word_index + 1, len(curr_paragraph.words)):
        word = curr_paragraph.words[i]
        for j in range(len(word.symbols)):  # start from next symbol
            symbol = word.symbols[j]
            if symbol.property.detected_break:
                if str(symbol.property.detected_break.type_) != "BreakType.SPACE":
                    return symbol.bounding_box

    # search through other paragraphs
    for par_idx in range(paragraph_index + 1, len(block.paragraphs)):
        paragraph = block.paragraphs[par_idx]
        for i in range(len(paragraph.words)):
            word = paragraph.words[i]
            for j in range(len(word.symbols)):  # start from next symbol
                symbol = word.symbols[j]
                if symbol.property.detected_break:
                    if str(symbol.property.detected_break.type_) != "BreakType.SPACE":
                        return symbol.bounding_box


def are_line_breaks_too_close(current_bounding_box, next_bounding_box):
    """
    if adjacent line breaks have similar y values, return False, i.e.
    treat it as a false positive

    :param current_bounding_box:
    :param next_bounding_box:
    :return:
    """
    y_diff = abs(current_bounding_box.vertices[0].y - next_bounding_box.vertices[0].y)
    if y_diff < 10:
        return True
    return False


def detect_document(path):
    """Detects document features in an image."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)

    return response.text_annotations[0].description if len(response.text_annotations) > 0 else []


def resize_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    # result = detect_document('static/text2.jpg')
    # print(result)
    # render_doc_text("static/text3.jpg", "static/bla.jpg")
    image_path_base = "static/text1"
    image_path = image_path_base + ".jpg"
    cv_image = cv2.imread(image_path)
    lines = find_lines(image_path)
    print(lines)
    print("Number of lines in image", image_path, ":", len(lines))
    
    # if new_img.shape[0] > 1000 or new_img.shape[1] > 1000:
    #     new_img = resize_image(new_img, 30)
    new_img = cv2.imread('static/uploaded_img_res.jpg')
    # cv2.imshow('window', new_img)
    # cv2.waitKey(0)
