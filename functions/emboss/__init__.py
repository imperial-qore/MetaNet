import logging
import azure.functions as func
from PIL import Image
from PIL import ImageFilter
import cv2 as cv
import numpy as np
import io

def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    body = req.get_body()

    try:
        image = Image.open(io.BytesIO(body))
    except IOError:
        return func.HttpResponse(
                "Bad input. Unable to cast request body to an image format.",
                status_code=400
        )

    result = run_inference(image, context)

    return func.HttpResponse(result)

def run_inference(image, context):
    # Perform filter operation
    original_image_size = image.size[0], image.size[1]
    result = image.filter(ImageFilter.EMBOSS)

    # Postprocess image
    img = result
    max_width  = 800
    height = int(max_width * original_image_size[1] / original_image_size[0])
    # Upsample and correct aspect ratio for final image
    img = img.resize((max_width, height), Image.BICUBIC)
    
    # Store inferred image as in memory byte array
    img_byte_arr = io.BytesIO()
    # Convert composite to RGB so we can return JPEG
    img.convert('RGB').save(img_byte_arr, format='JPEG')
    final_image = img_byte_arr.getvalue()

    return final_image