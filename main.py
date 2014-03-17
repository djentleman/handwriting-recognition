# handwriting recognition - remote octave calls
from PIL import Image
import pygame
import pygame.camera as camera
import math
import subprocess
import time


def call_octave(data):
    out = subprocess.Popen("octave\octave_call.bat " + data, shell=1, stdout=subprocess.PIPE)
    out_data = out.communicate()[0].replace("\r", "").split("\n")
    out_vector = (out_data[-7] + out_data[-3]).split("  ")[1:]
    for i in range(len(out_vector)):
        out_vector[i] = float(out_vector[i])
    return out_vector
    



def init_cam():
    pygame.init()
    camera.init()
    webcam = camera.Camera(camera.list_cameras()[0])
    return webcam

def get_image(webcam, show=False):
    img = webcam.get_image()
    pygame.image.save(img, "current.png")
    img = Image.open("current.png")
    if show:
        img.show()
    return img

def process_image(img, show=False):
    image_scale = 10
    img = img.convert('L')
    # scale down to something manageable by neural net, 20x20
    img = img.resize((image_scale,image_scale), Image.ANTIALIAS)
    if show:
        img.show()
    return img

def get_dataset(img):
    image_scale = 10
    rows = []
    cols = []
    for i in range(image_scale):
        cols.append(1 / (1 + (math.e ** -(sum([img.getpixel((i, pix)) - (255 / 2) for pix in range(image_scale)]) / (image_scale * 20.0)))))
        rows.append(1 / (1 + (math.e ** -(sum([img.getpixel((pix, i)) - (255 / 2) for pix in range(image_scale)]) / (image_scale * 20.0)))))

    return cols + rows

def stringify(dataset):
    dump = ""
    for element in dataset:
        dump += '{0:.3g}'.format(element) + " "
    return dump[:-1]

def event_loop(cam):
    while 1:
        img = process_image(get_image(cam))
        dataset = stringify(get_dataset(img))
        vector = call_octave(dataset)
        print vector
        print "prediction: %d"%(vector.index(max(vector)))
        time.sleep(0.3)
        

def main():
        cam = init_cam()
        event_loop(cam)
        print "No Camera Found..."
        return

if __name__ == "__main__":
    main()
