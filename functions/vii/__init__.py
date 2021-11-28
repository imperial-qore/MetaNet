import logging
import azure.functions as func
from PIL import Image
import numpy as np
import io
import cv2
from time import sleep
import imutils

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
    # Save PIL image
    image.save(f'{context.function_directory}/temp.jpg')

    # Load Model
    model_path = f'{context.function_directory}/vii.t7'
    net = cv2.dnn.readNetFromTorch(model_path)

    # Preprocess image
    image = cv2.imread(f'{context.function_directory}/temp.jpg')
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # Process image
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h),
        (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()

    # reshape the output tensor, add back in the mean subtraction, and
    # then swap the channel ordering
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output = output.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(output)
    sleep(2)
    
    # Store inferred image as in memory byte array
    img_byte_arr = io.BytesIO()
    # Convert composite to RGB so we can return JPEG
    img.convert('RGB').save(img_byte_arr, format='JPEG')
    final_image = img_byte_arr.getvalue()

    return final_image