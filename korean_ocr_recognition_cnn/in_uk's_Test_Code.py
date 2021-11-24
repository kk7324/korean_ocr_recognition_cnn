#!/usr/bin/env python

import argparse
import io
import os
import sys

import tensorflow as tf

''' DEFINE PARAMETER '''
# 사용한 테스트 폰트의 종류
font_count = 20

# generator.py 에서 사용한 Distortion Count -> (원본1 + Distortion Count)씩 이미지가 생성될것이다.
distortion_count = 0

# 테스트 하고자 하는 이미지의 총 갯수
# total_test_image = 5
total_test_image = 512 * font_count * (distortion_count + 1)

# 생성될 총 테스트 이미지의 갯수는 font_count * (distortion_count + 1) * 512일 것이다.

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default paths.

DEFAULT_LABEL_FILE = os.path.join(
    SCRIPT_PATH, 'C:/Users/stork/Desktop/hangul/tensorflow-hangul-recognition-master/labels/512-common-hangul.txt'
)
DEFAULT_GRAPH_FILE = os.path.join(
    SCRIPT_PATH, 'C:/Users/stork/Desktop/hangul/tensorflow-hangul-recognition-master/랜덤필터적용2/랜덤필터적용2/optimized_hangul_tensorflow.pb'
)

DEFAULT_TEST_LABEL_FILE = os.path.join(
    SCRIPT_PATH, 'C:/Users/stork/Desktop/hangul/tensorflow-hangul-recognition-master/labels/512-common-hangul.txt'
)





def read_image(file):
    """Read an image file and convert it into a 1-D floating point array."""
    file_content = tf.read_file(file)
    image = tf.image.decode_jpeg(file_content, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.reshape(image, (1, 64*64))
    return image


def classify(args):
    """Classify a character.

    This method will import the saved model from the given graph file, and will
    pass in the given image pixels as input for the classification. The top
    five predictions will be printed.
    """
    # 테스트 하고자 하는 입력 이미지의 총 갯수
    image_count = total_test_image



    default_input_image_path = "C:/Users/stork/Desktop/hangul/tensorflow-hangul-recognition-master/20fontimage/hangul-images"


    cnt = 0

    error_count = 0

    for i in range(image_count):
        cnt = cnt + 1
        labels = io.open(args.label_file,
                         'r', encoding='utf-8').read().splitlines()

        # 테스트 입력 이미지의 라벨 파일(test_label.txt)
        answer_labels = io.open(args.test_label,
                                'r', encoding='utf-8').read().splitlines()

        input_image_path = default_input_image_path + "/" + str(i) + ".jpeg"


        if not os.path.isfile(input_image_path):
            print('Error: Image %s not found.' % input_image_path)
            sys.exit(1)

        # Load graph and parse file.
        with tf.gfile.GFile(args.graph_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name='hangul-model',
                producer_op_list=None
            )

        # Get relevant nodes.
        x = graph.get_tensor_by_name('hangul-model/input:0')
        y = graph.get_tensor_by_name('hangul-model/output:0')

        image = read_image(input_image_path)

        sess = tf.InteractiveSession()
        image_array = sess.run(image)
        sess.close()
        with tf.Session(graph=graph) as graph_sess:
            predictions = graph_sess.run(y, feed_dict={x: image_array})
            prediction = predictions[0]

        # Get the indices that would sort the array, then only get the indices that
        # correspond to the top 5 predictions.
        sorted_indices = prediction.argsort()[::-1][:5]

        # label : 모델에서 예측한 정답을 담은 변수
        # labels : 512-common-haugul.txt 이 파일
        label = labels[sorted_indices[0]]
        confidence = prediction[sorted_indices[0]]

        val = font_count * (distortion_count + 1)
        new_index = (i // val)


        if (label != answer_labels[new_index]):
            error_count = error_count + 1
            current_error_rate = error_count / cnt * 100
            print('예측값(오답) : %s  /  예측 확률 : %s  /  정답 : %s' % (label, confidence, answer_labels[new_index]))
            print('현재까지 에러률 : ', current_error_rate, " %")
            print('파일 index number : ', i)


        '''
        if(label != answer_labels[i]):
            print('예측값(오답) : %s  /  예측 확률 : %s  /  정답 : %s' %(label, confidence, answer_labels[i]))
            print('파일 index number : ', i)

        '''




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')

    parser.add_argument('--graph-file', type=str, dest='graph_file',
                        default=DEFAULT_GRAPH_FILE,
                        help='The saved model graph file to use for '
                             'classification.')

    parser.add_argument('--test-label', type=str, dest='test_label',
                        default=DEFAULT_TEST_LABEL_FILE,
                        help='테스트를 하기위한 이미지의 라벨파일의 경로')
    classify(parser.parse_args())