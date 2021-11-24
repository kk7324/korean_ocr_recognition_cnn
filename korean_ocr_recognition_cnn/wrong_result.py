#!/usr/bin/env python

import argparse
import glob
import io
import os
import random

import numpy
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,'C:/Users/stork/Desktop/hangul/tensorflow-hangul-recognition-master/labels/512-common-hangul.txt')
DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, 'C:/Users/stork/Desktop/hangul/tensorflow-hangul-recognition-master/font_test3')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, 'C:/Users/stork/Desktop/hangul/tensorflow-hangul-recognition-master/compareimagetriple_east1')

# Number of random distortion images to generate per font and character.
DISTORTION_COUNT = 0

# Width and height of the resulting image.
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64


def compare_index(index):
### 오답 index 넣는 부분
    a_n = {51,52,77,82,83,101,112,119,122,123,153,229,237,244,248,249,262,267,271,275,278,320,322,323,324,333,336,362,373,382,383,440,441,454,455,465,475,533,552,553,554,555,581,583,599,612,693,762,765,805,813,815,895,952,961,963,976,1019,1036,1079,1152,1212,1233,1264,1320,1321,1333,1342,1349,1384,1393,1442,1461,1475,1480,1481,1482,1485,1487,1488,1489,1491,1493,1498,1499,1522,1545,1553,1561,1605,1612,1613,1620,1632,1676,1683,1781,1832,1873,1879,1893,1932,1933,1976,1992,2003,2016,2076,2085,2095,2160,2161,2163,2173,2176,2184,2204,2239,2260,2274,2276,2304,2353,2358,2374,2392,2400,2401,2402,2403,2444,2451,2453,2456,2473,2493,2500,2505,2506,2533,2557,2572,2632,2633,2639,2796,2812,2900,2901,2913,2916,2983,2985,3052,3062,3079,3104,3122,3124,3136,3176,3191,3196,3212,3272,3324,3332,3356,3372,3376,3379,3451,3465,3471,3473,3532,3533,3572,3633,3749,3752,3753,3772,3773,3775,3777,3802,3805,3853,3876,3913,3932,3938,3963,3979,3982,3992,4033,4035,4053,4083,4092,4116,4156,4179,4193,4236,4272,4293,4312,4319,4362,4372,4373,4399,4412,4418,4436,4452,4516,4550,4576,4677,4736,4742,4745,4752,4755,4779,4796,4820,4859,4913,4939,4972,4984,5033,5081,5092,5156,5161,5163,5173,5193,5272,5341,5392,5412,5419,5459,5473,5489,5621,5692,5765,5775,5777,5801,5804,5813,5840,5856,5869,5870,5873,5878,5884,5897,5917,5919,5972,5980,5981,5983,5987,5996,5999,6036,6052,6100,6131,6154,6160,6227,6293,6401,6405,6433,6444,6445,6459,6536,6604,6612,6613,6633,6696,6700,6707,6711,6718,6817,6822,6825,6835,6853,6866,6894,6899,6901,6905,6924,6952,6992,7013,7055,7056,7059,7084,7093,7096,7160,7201,7204,7207,7208,7209,7210,7211,7216,7232,7433,7499,7513,7516,7540,7541,7542,7543,7556,7583,7597,7665,7833,7856,7873,7903,7913,7919,7922,8011,8012,8013,8040,8041,8042,8049,8050,8059,8063,8072,8092,8132,8133,8264,8319,8324,8336,8352,8373,8416,8452,8472,8473,8492,8496,8532,8533,8562,8612,8618,8676,8679,8693,8721,8722,8739,8752,8773,8792,8793,8820,8823,8839,8853,8913,8949,8960,8973,8996,9002,9011,9012,9025,9032,9172,9173,9219,9245,9421,9423,9424,9452,9453,9505,9602,9604,9705,9712,9733,9813,9873,9904,9912,9977,10013,10045,10065,10072,10100,10143,10144,10173,10202}
    result=0
    index=int(index)
    for i in a_n:
        if i == index:
            result= 1
    return result

def generate_hangul_images(label_file, fonts_dir, output_dir):
    """Generate Hangul image files.

    This will take in the passed in labels file and will generate several
    images using the font files provided in the font directory. The font
    directory is expected to be populated with *.ttf (True Type Font) files.
    The generated images will be stored in the given output directory. Image
    paths will have their corresponding labels listed in a CSV file.
    """
    with io.open(label_file, 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()

    image_dir = os.path.join(output_dir, 'hangul-images')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    # Get a list of the fonts.
    fonts = glob.glob(os.path.join(fonts_dir, '*.ttf'))

    labels_csv = io.open(os.path.join(output_dir, 'labels-map.csv'), 'w',

                         encoding='utf-8')

    total_count = -1
    prev_count = 0
    for character in labels:
        # Print image count roughly every 5000 images.
        if total_count - prev_count > 500:
            prev_count = total_count
            print('{} images generated...'.format(total_count))

        for font in fonts:
            total_count += 1
            ox=0
            ox=compare_index(total_count)
            if ox == 1:
                image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color='white')
                #image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=0)
                font = ImageFont.truetype(font, 48)
                drawing = ImageDraw.Draw(image)
                w, h = drawing.textsize(character, font=font)
                drawing.text(
                    ((IMAGE_WIDTH-w)/2, (IMAGE_HEIGHT-h)/2),
                    character,
                    fill='black',
                    #fill=(255),
                    font=font
                )
                file_string = '{}.jpeg'.format(total_count)
                file_path = os.path.join(image_dir, file_string)
                image.save(file_path, 'JPEG')
#                labels_csv.write(u'{},{}\n'.format(file_path, character))
                labels_csv.write('{}\n'.format(character))

                for i in range(DISTORTION_COUNT):
                    total_count += 1
                    file_string = '{}.jpeg'.format(total_count)
                    file_path = os.path.join(image_dir, file_string)
                    arr = numpy.array(image)

                    distorted_array = elastic_distort(
                        arr, alpha=random.randint(30, 36),
                        sigma=random.randint(5, 6)
                    )
                    distorted_image = Image.fromarray(distorted_array)
                    distorted_image.save(file_path, 'JPEG')
                    labels_csv.write('{}\n'.format(character))

    print('Finished generating {} images.'.format(total_count))
    labels_csv.close()


def elastic_distort(image, alpha, sigma):
    """Perform elastic distortion on an image.

    Here, alpha refers to the scaling factor that controls the intensity of the
    deformation. The sigma variable refers to the Gaussian filter standard
    deviation.
    """
    random_state = numpy.random.RandomState(None)
    shape = image.shape

    dx = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha
    dy = gaussian_filter(
        (random_state.rand(*shape) * 2 - 1),
        sigma, mode="constant"
    ) * alpha

    x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
    indices = numpy.reshape(y+dy, (-1, 1)), numpy.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, mode='nearest', order=1).reshape(shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--font-dir', type=str, dest='fonts_dir',
                        default=DEFAULT_FONTS_DIR,
                        help='Directory of ttf fonts to use.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store generated images and '
                             'label CSV file.')
    args = parser.parse_args()
    generate_hangul_images(args.label_file, args.fonts_dir, args.output_dir)