import uuid
import cv2

from detect.detect import Detector


detector = Detector()

def upload_image(filename):
    print('uploading image...')

def main():

    camera = cv2.VideoCapture(0)
    _, image = camera.read()
    image = cv2.imread('images/car-1.jpg')
    output = detector.detect(image)

    id = str(uuid.uuid4())
    filename = f"processed_images/{id}.jpg"
    cv2.imwrite(filename,output.image)
    upload_image(filename)


main()