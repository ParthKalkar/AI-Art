'''
Genetic algorithm by Parth Kalkar
For running code we need to set name of initial image, 
quality of result and maximum circle radius.
Kind of
>> python3 genetic_algorithm.py 3.png 95 10
'''
from copy import deepcopy
import random
import sys
import math
import re
from typing import Tuple

from PIL import Image
from PIL import ImageDraw
import logging
import datetime
from numba import jit, types, int64, njit
import numpy as np
from numba.experimental import jitclass

BLACK_BACKGROUND = (0, 0, 0, 255)
MAX_CIRCLE_RADIUS = int(sys.argv[3])
SIZE = 512
NUMBER_OF_CIRCLES = math.floor((SIZE**2))
QUALITY = int(sys.argv[2])
INPUT_NAME = str(sys.argv[1])
OUTPUT_NAME = "result_"+'radius=' + \
    str(MAX_CIRCLE_RADIUS)+'_'+'quality='+str(QUALITY)+'_'+str(sys.argv[1])
OUTPUT_DIRECTORY = "results/"
INPUT_DIRECTORY = "data/"
LOGS = "logs_"+'radius=' + \
    str(MAX_CIRCLE_RADIUS)+'_'+'quality='+str(QUALITY) + '_circle_number='+str(NUMBER_OF_CIRCLES) + \
    '_'+re.split(r'png', str(sys.argv[1]))[0] + 'log'


# Made circles with color, location and radius.
@jitclass([('color', types.UniTuple(int64, 3))])
class Circle(object):
    color: Tuple
    x: int
    y: int
    radius: int

    def __init__(self, color, x, y, radius):
        self.color = color
        self.x = x
        self.y = y
        self.radius = radius

    def set_new_color(self, new_color):
        self.color = tuple(new_color)

    def copy(self):
        return Circle(self.color, self.x, self.y, self.radius)


# Load initial image.
def load_image():
    img = Image.open(INPUT_DIRECTORY+INPUT_NAME)
    img = img.convert('RGB')
    logging.info('Load image '+INPUT_NAME)
    return img


# Generate random location.
@njit
def generate_point():
    x = random.randrange(0, SIZE)
    y = random.randrange(0, SIZE)
    return x, y


# Generate random color.
@njit
def generate_color():
    red = random.randrange(0, 256)
    green = random.randrange(0, 256)
    blue = random.randrange(0, 256)
    return red, green, blue


# Generate list of circles.
@njit
def generate_circles(number_of_circles):
    circles = []
    for i in range(0, number_of_circles):
        x, y = generate_point()
        color = generate_color()
        radius = random.randrange(0, MAX_CIRCLE_RADIUS)
        circle = Circle(color, x, y, radius)
        circles.append(circle)
    return circles


# Get average color of polygon 3x3, where (x,y) is a centre of polygon.
@njit
def get_average_color(image, x, y):
    pixel_color = [image[x + i + (y + j)*SIZE]for i in (-1, 0, 1) for j in (-1, 0, 1)
                   if 0 <= x + i <= SIZE - 1 and 0 <= y + j <= SIZE - 1]
    sum_red = 0
    sum_green = 0
    sum_blue = 0
    for (red, green, blue) in pixel_color:
        sum_red += red
        sum_green += green
        sum_blue += blue

    red = math.floor(sum_red/len(pixel_color))
    green = math.floor(sum_green/len(pixel_color))
    blue = math.floor(sum_blue/len(pixel_color))

    return red, green, blue


# Compute fitness function for colors.
@njit
def fitness_function(initial_color, new_color):
    max_diff = 441
    r_1, g_1, b_1 = initial_color
    r_2, g_2, b_2 = new_color

    pixel_diff = math.sqrt((r_1 - r_2)**2 + (g_1 - g_2)**2 + (b_1 - b_2)**2)
    fit = (1 - pixel_diff/max_diff) * 100
    return fit


# Draw circle on canvas.
def draw_circle(canvas, circle):
    leftUpPoint = (circle.x-circle.radius, circle.y-circle.radius)
    rightDownPoint = (circle.x+circle.radius, circle.y+circle.radius)
    twoPointList = [leftUpPoint, rightDownPoint]
    canvas.ellipse(twoPointList, fill=tuple(circle.color))

    return canvas


def main():
    # Base desription logs.
    logging.basicConfig(filename=LOGS, level=logging.DEBUG)
    logging.info('Start time: ' +
                 str(datetime.datetime.now().time().strftime('%H:%M:%S')))
    logging.info('Take picture: '+str(INPUT_NAME))
    logging.info('Quality: '+str(QUALITY)+' %')
    logging.info('Number of circles: '+str(NUMBER_OF_CIRCLES))
    logging.info('Max radius of circles: '+str(MAX_CIRCLE_RADIUS)+' pixels')

    # Load image.
    image: Image = load_image()
    data = np.array(image.getdata())
    img_width, img_height = image.size
    if img_width != SIZE or img_height != SIZE:
        sys.exit('Size of input image out of range.')
        logging.error('Size of input image out of range. Not 512x512.')

    # Made clear black canvas.
    canvas = Image.new('RGB', (SIZE, SIZE), color=BLACK_BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    # Made list of circles.
    number_of_circles = NUMBER_OF_CIRCLES
    circles = generate_circles(number_of_circles)

    # Start generation.
    iteration = 0
    circle_number = 0
    for circle in circles:
        pixel_color = get_average_color(data, circle.x, circle.y)

        iteration += 1
        circle_number += 1
        # Get fitness value.
        iteration = fitness_value(circle, iteration, pixel_color)

        # Add circle on canvas.
        draw = draw_circle(draw, circle)
        logging.info('Position of circle: '+str(circle_number) +
                     ' Number of childs: '+str(iteration))


    # Save result image.
    canvas.resize((512, 512)).save(OUTPUT_DIRECTORY+OUTPUT_NAME)
    logging.info('Save image to '+str(OUTPUT_DIRECTORY+OUTPUT_NAME))

    logging.info(
        'End time: '+str(datetime.datetime.now().time().strftime('%H:%M:%S')))
    return sys.exit(0)


@jit(nopython=True)
def fitness_value(circle, iteration, pixel_color):
    fitness_coef = fitness_function(pixel_color, circle.color)
    while fitness_coef < QUALITY:

        child_circle = circle.copy()
        child_circle.set_new_color(generate_color())

        child_fitness = fitness_function(pixel_color, child_circle.color)
        if child_fitness > fitness_coef:
            circle.set_new_color(child_circle.color)
            fitness_coef = child_fitness
        iteration += 1
    return iteration


if __name__ == "__main__":
    main()
