from PIL import Image
from pytesseract import *

def OCR(imgfile):
    im = Image.open(imgfile)
    text = image_to_string(im, lang='kor')

    print(text)

OCR('3.png')
