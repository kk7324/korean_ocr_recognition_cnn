#!/usr/bin/env python

import argparse
import io
import os
import numpy as np
import tensorflow as tf

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,'C:/Users/stork/Desktop/hangul/tensorflow-hangul-recognition-master/labels/512-common-hangul.txt')
DEFAULT_TFRECORDS_DIR = os.path.join(SCRIPT_PATH, 'C:/Users/stork/Desktop/hangul/tensorflow-hangul-recognition-master/tfrecords-output-1108')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, 'C:/Users/stork/Desktop/hangul/tensorflow-hangul-recognition-master/saved-model-1108')

MODEL_NAME = 'hangul_tensorflow'
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

DEFAULT_NUM_EPOCHS = 15
BATCH_SIZE = 100

# This will be determined by the number of entries in the given label file.
num_classes = 512


def _parse_function(example):
    features = tf.parse_single_example(
        example,
        features={
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                                default_value='')
        })
    label = features['image/class/label']
    image_encoded = features['image/encoded']

    # Decode the JPEG.
    image = tf.image.decode_jpeg(image_encoded, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.reshape(image, [IMAGE_WIDTH*IMAGE_HEIGHT])

    # Represent the label as a one hot vector.
    label = tf.stack(tf.one_hot(label, num_classes))
    return image, label


def export_model(model_output_dir, input_node_names, output_node_name):
    """Export the model so we can use it later.

    This will create two Protocol Buffer files in the model output directory.
    These files represent a serialized version of our model with all the
    learned weights and biases. One of the ProtoBuf files is a version
    optimized for inference-only usage.
    """

    name_base = os.path.join(model_output_dir, MODEL_NAME)
    frozen_graph_file = os.path.join(model_output_dir,
                                     'frozen_' + MODEL_NAME + '.pb')
    freeze_graph.freeze_graph(
        name_base + '.pbtxt', None, False, name_base + '.chkp',
        output_node_name, "save/restore_all", "save/Const:0",
        frozen_graph_file, True, ""
    )

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(frozen_graph_file, "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, [input_node_names], [output_node_name],
            tf.float32.as_datatype_enum)

    optimized_graph_file = os.path.join(model_output_dir,
                                        'optimized_' + MODEL_NAME + '.pb')
    with tf.gfile.GFile(optimized_graph_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("Inference optimized graph saved at: " + optimized_graph_file)


def weight_variable(shape):
    """Generates a weight variable of a given shape."""
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weight')


def bias_variable(shape):
    """Generates a bias variable of a given shape."""
    initial = tf.constant(0, shape=shape,dtype=float)
    return tf.Variable(initial, name='bias')


def main(label_file, tfrecords_dir, model_output_dir, num_train_epochs):
    """Perform graph definition and model training.

    Here we will first create our input pipeline for reading in TFRecords
    files and producing random batches of images and labels.
    Next, a convolutional neural network is defined, and training is performed.
    After training, the model is exported to be used in applications.
    """
    global num_classes
    labels = io.open(label_file, 'r', encoding='utf-8').read().splitlines()
    num_classes = len(labels)

    # Define names so we can later reference specific nodes for when we use
    # the model for inference later.
    input_node_name = 'input'
    keep_prob_node_name = 'keep_prob'
    output_node_name = 'output'

    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    print('Processing data...')

    tf_record_pattern = os.path.join(tfrecords_dir, '%s-*' % 'train')
    train_data_files = tf.gfile.Glob(tf_record_pattern)

    tf_record_pattern = os.path.join(tfrecords_dir, '%s-*' % 'test')
    test_data_files = tf.gfile.Glob(tf_record_pattern)

    # Create training dataset input pipeline.
    train_dataset = tf.data.TFRecordDataset(train_data_files) \
        .map(_parse_function) \
        .shuffle(1000) \
        .repeat(num_train_epochs) \
        .batch(BATCH_SIZE) \
        .prefetch(1)

    # Create the model!

    # Placeholder to feed in image data.
    x = tf.placeholder(tf.float32, [None, IMAGE_WIDTH*IMAGE_HEIGHT],
                       name=input_node_name)
    # Placeholder to feed in label data. Labels are represented as one_hot
    # vectors.
    y_ = tf.placeholder(tf.float32, [None, num_classes])

    # Reshape the image back into two dimensions so we can perform convolution.
    x_image = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])

    # First convolutional layer. 32 feature maps.
    ###이 부분 우리의 필터 값으로 초기화 해주는 과정
    #W_conv1 = weight_variable([5, 5, 1, 32])

    x1 = np.array(
        [0.2504,0.2240,0.1412,0.1016,0.1035,0.2993,0.2899,0.1525,0.0527,0.0621,0.3689,0.4216,0.1769,0.0001,0.0471,0.3200,0.2974,0.1487,0.0433,0.0621,0.2541,0.2089,0.1186,0.0715,0.0847])
    x2 = np.array(
        [0.0828,0.1016,0.1656,0.2993,0.3294,0.0471,0.0565,0.1600,0.3464,0.3689,0.0414,0.0001,0.1506,0.3784,0.3520,0.0565,0.0339,0.1035,0.2353,0.2692,0.1035,0.0791,0.1129,0.1864,0.2296])
    x3 = np.array(
        [0.0621,0.0527,0.0320,0.0282,0.0282,0.0621,0.0452,0.0001,0.0414,0.0715,0.1336,0.1600,0.1600,0.1619,0.1544,0.2974,0.3652,0.4216,0.3539,0.2918,0.3972,0.4216,0.4179,0.3821,0.3332])
    x4 = np.array(
        [0.3087,0.3934,0.4216,0.4367,0.4009,0.2767,0.3727,0.4367,0.3765,0.3106,0.1581,0.1864,0.1769,0.1751,0.1656,0.0828,0.0452,0.0001,0.0395,0.0678,0.0376,0.0226,0.0282,0.0546,0.0828])
    x5 = np.array(
        [0.0866,0.0659,0.0678,0.0640,0.0546,0.1525,0.1148,0.0489,0.0001,0.0489,0.2014,0.2259,0.1449,0.0809,0.0809,0.3049,0.3708,0.2786,0.1920,0.1280,0.3708,0.3576,0.3313,0.2880,0.2052])
    x6 = np.array(
        [0.1826,0.2598,0.3181,0.3708,0.3934,0.1242,0.2089,0.3332,0.4442,0.3765,0.0565,0.0809,0.1732,0.2824,0.2711,0.0508,0.0132,0.0734,0.1506,0.1920,0.0621,0.0584,0.0659,0.0772,0.1148])
    x7 = np.array(
        [0.0339,0.0508,0.0866,0.1299,0.1713,0.0471,0.0001,0.0772,0.1751,0.2146,0.0866,0.0772,0.1600,0.2579,0.2409,0.1186,0.1506,0.2391,0.3464,0.2805,0.1562,0.1845,0.2202,0.2635,0.2880])
    x8 = np.array(
        [0.3576,0.3106,0.2635,0.2654,0.1242,0.3520,0.3953,0.2767,0.1525,0.0734,0.3087,0.2936,0.1694,0.0678,0.0565,0.2409,0.1751,0.0715,0.0001,0.0546,0.1638,0.1054,0.0753,0.0640,0.0696])

    r1 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r2 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r3 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r4 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r5 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r6 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r7 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r8 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r9 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r10 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r11 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r12 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r13 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r14 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r15 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r16 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r17 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r18 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r19 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r20 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r21 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r22 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r23 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    r24 = tf.random.truncated_normal([5, 5, 1], stddev=0.1)
    # r25= tf.random.truncated_normal([5,5,1], stddev=0.1)
    # r26= tf.random.truncated_normal([5,5,1], stddev=0.1)
    # r27= tf.random.truncated_normal([5,5,1], stddev=0.1)
    # r28= tf.random.truncated_normal([5,5,1], stddev=0.1)
    k1 = np.reshape(x1, (5, 5, 1))
    k2 = np.reshape(x2, (5, 5, 1))
    k3 = np.reshape(x3, (5, 5, 1))
    k4 = np.reshape(x4, (5, 5, 1))
    k5 = np.reshape(x5, (5, 5, 1))
    k6 = np.reshape(x6, (5, 5, 1))
    k7 = np.reshape(x7, (5, 5, 1))
    k8 = np.reshape(x8, (5, 5, 1))
    z1 = tf.Variable(k1, dtype=float)
    z2 = tf.Variable(k2, dtype=float)
    z3 = tf.Variable(k3, dtype=float)
    z4 = tf.Variable(k4, dtype=float)
    z5 = tf.Variable(k5, dtype=float)
    z6 = tf.Variable(k6, dtype=float)
    z7 = tf.Variable(k7, dtype=float)
    z8 = tf.Variable(k8, dtype=float)


    z = tf.stack(
        [z1, z2, z3, z4, z5, z6, z7, z8, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17,
         r18, r19, r20, r21, r22, r23, r24], axis=3)
    # 수정완료
    W_conv1 = tf.Variable(z, name='weight')


    #W_conv1 = tf.stack([r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32], axis=3)
    #W_conv1 = tf.Variable(tf.stack([r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, r32],
    #                              axis=3), name='weight')


    b_conv1 = bias_variable([32])
    x_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1],
                           padding='SAME')

    # 원본 코드
    h_conv1 = tf.nn.relu(x_conv1 + b_conv1)
    # 수정된 코드, Activation function = Identity
    #h_conv1 = tf.identity(x_conv1 + b_conv1)

    # Max-pooling.
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')
    # 에러 발생시 이부분 수정
    # 추가된 코드, Activation function = ReLu
    #h_pool1 = tf.nn.relu(h_pool1)

    # Second convolutional layer. 64 feature maps.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    x_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1],
                           padding='SAME')
    # 원본 코드
    h_conv2 = tf.nn.relu(x_conv2 + b_conv2)
    # 수정된 코드, Activation function = Identity
    #h_conv2 = tf.identity(x_conv2 + b_conv2)

    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')
    # 에러 발생시 이부분 수정
    # 추가된 코드, Activation function = ReLu
    #h_pool2 = tf.nn.relu(h_pool2)

    # Third convolutional layer. 128 feature maps.
    W_conv3 = weight_variable([4, 4, 64, 128])
    b_conv3 = bias_variable([128])
    x_conv3 = tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1],
                           padding='SAME')
    # 원본 코드
    h_conv3 = tf.nn.relu(x_conv3 + b_conv3)
    # 수정된 코드, Activation function = Identity
    #h_conv3 = tf.identity(x_conv3 + b_conv3)

    h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')
    # 에러 발생시 이부분 수정
    # 추가된 코드, Activation function = ReLu
    #h_pool3 = tf.nn.relu(h_pool3)

    """ 새롭게 추가하는 은닉층 """
    # 4th convolutional layer. 256 feature maps.
    W_conv4 = weight_variable([4, 4, 128, 256])
    b_conv4 = bias_variable([256])
    x_conv4 = tf.nn.conv2d(h_pool3, W_conv4, strides=[1, 1, 1, 1],
                           padding='SAME')

    """ 이 부분 조심 파라미터를 그대로 사용하면 안됬었음"""
    h_conv4 = tf.nn.relu(x_conv4 + b_conv4)
    # 수정된 코드, Activation function = Identity
    #h_conv4 = tf.identity(x_conv4 + b_conv4)

    h_pool4 = tf.nn.max_pool(h_conv4, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')
    # 추가된 코드, Activation function = ReLu
    #h_pool4 = tf.nn.relu(h_pool4)

    # Fully connected layer. Here we choose to have 384 neurons in this layer.
    W_fc1 = weight_variable([4 * 4 * 256, 384])
    b_fc1 = bias_variable([384])
    h_pool_flat = tf.reshape(h_pool4, [-1, 4 * 4 * 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)


    '''
    # Dropout layer. This helps fight overfitting.
    keep_prob = tf.placeholder(tf.float32, name=keep_prob_node_name)
    h_fc1_drop = tf.nn.dropout(h_fc1, rate=1-keep_prob)
    '''


    # Classification layer(Fully connected layer).
    W_fc2 = weight_variable([384, num_classes])
    b_fc2 = bias_variable([num_classes])

    #y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    # Dropout 계층 비활성화
    y = tf.matmul(h_fc1, W_fc2) + b_fc2

    # This isn't used for training, but for when using the saved model.
    tf.nn.softmax(y, name=output_node_name)

    # Define our loss.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(y_),
            logits=y
        )
    )



    # Define our optimizer for minimizing our loss. Here we choose a learning
    # rate of 0.0001 with AdamOptimizer. This utilizes something
    # called the Adam algorithm, and utilizes adaptive learning rates and
    # momentum to get past saddle points.
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

    # Define accuracy.
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)



    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize the variables.
        sess.run(tf.global_variables_initializer())

        checkpoint_file = os.path.join(model_output_dir, MODEL_NAME + '.chkp')

        # Save the graph definition to a file.
        tf.train.write_graph(sess.graph_def, model_output_dir,
                             MODEL_NAME + '.pbtxt', True)

        try:
            iterator = train_dataset.make_one_shot_iterator()
            batch = iterator.get_next()
            step = 0

            while True:

                # Get a batch of images and their corresponding labels.
                train_images, train_labels = sess.run(batch)

                # Perform the training step, feeding in the batches.
                '''
                sess.run(train_step, feed_dict={x: train_images,
                                                y_: train_labels,
                                                keep_prob: 0.5})
                '''
                sess.run(train_step, feed_dict={x: train_images,
                                                y_: train_labels})


                if step % 100 == 0:
                    train_accuracy = sess.run(
                        accuracy,
                        # feed_dict={x: train_images, y_: train_labels,
                        #                                    keep_prob: 1.0}
                        feed_dict = {x: train_images, y_: train_labels}
                    )
                    print("Step %d, Training Accuracy %g" %
                          (step, float(train_accuracy)))

                # Every 10,000 iterations, we save a checkpoint of the model.
                if step % 10000 == 0:
                    saver.save(sess, checkpoint_file, global_step=step)

                step += 1

        except tf.errors.OutOfRangeError:
            pass

        # Save a checkpoint after training has completed.
        saver.save(sess, checkpoint_file)

        # See how model did by running the testing set through the model.
        print('Testing model...')

        # Create testing dataset input pipeline.
        test_dataset = tf.data.TFRecordDataset(test_data_files) \
            .map(_parse_function) \
            .batch(BATCH_SIZE) \
            .prefetch(1)

        # Define a different tensor operation for summing the correct
        # predictions.
        accuracy2 = tf.reduce_sum(correct_prediction)
        total_correct_preds = 0
        total_preds = 0

        try:
            iterator = test_dataset.make_one_shot_iterator()
            batch = iterator.get_next()
            while True:
                test_images, test_labels = sess.run(batch)
                # acc = sess.run(accuracy2, feed_dict={x: test_images,
                #                                                      y_: test_labels,
                #                                                      keep_prob: 1.0})
                acc = sess.run(accuracy2, feed_dict={x: test_images,
                                                     y_: test_labels})
                total_preds += len(test_images)
                total_correct_preds += acc

        except tf.errors.OutOfRangeError:
            pass

        test_accuracy = total_correct_preds/total_preds
        print("Testing Accuracy {}".format(test_accuracy))

        export_model(model_output_dir, input_node_name,
                     output_node_name)

        sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--tfrecords-dir', type=str, dest='tfrecords_dir',
                        default=DEFAULT_TFRECORDS_DIR,
                        help='Directory of TFRecords files.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store saved model files.')
    parser.add_argument('--num-train-epochs', type=int,
                        dest='num_train_epochs',
                        default=DEFAULT_NUM_EPOCHS,
                        help='Number of times to iterate over all of the '
                             'training data.')
    args = parser.parse_args()
    main(args.label_file, args.tfrecords_dir,
         args.output_dir, args.num_train_epochs)