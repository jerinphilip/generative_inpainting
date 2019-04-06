import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from .inpaint_model import InpaintCAModel


parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')

class GInpWrapper:
    def __init__(self, checkpoint_dir):
        self.graph = tf.Graph()
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.session =  tf.Session(config=sess_config, graph=self.graph)
        # saver = tf.train.Saver()
        # checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
        # saver.restore(self.session, checkpoint_dir)
        with self.graph.as_default():
            self.model = InpaintCAModel()
            self.input_image = tf.placeholder(tf.float32, shape=(1, 256, 512, 3))
            self.output = self.model.build_server_graph(self.input_image)
            self.output = (self.output + 1.) * 127.5
            self.output = tf.reverse(self.output, [-1])
            self.output = tf.saturate_cast(self.output, tf.uint8)
            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            self.assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                var_value = tf.contrib.framework.load_variable(checkpoint_dir, from_name)
                self.assign_ops.append(tf.assign(var, var_value))
            self.session.run(self.assign_ops)
            print('Model loaded.')

    def predict(self, image, mask):
        with self.graph.as_default():
            image = cv2.imread(args.image)
            mask = cv2.imread(args.mask)
            assert image.shape == mask.shape
            h, w, _ = image.shape
            grid = 8
            image = image[:h//grid*grid, :w//grid*grid, :]
            mask = mask[:h//grid*grid, :w//grid*grid, :]
            print('Shape of image: {}'.format(image.shape))
            image = np.expand_dims(image, 0)
            mask = np.expand_dims(mask, 0)
            input_image = np.concatenate([image, mask], axis=2)
            # input_image = tf.constant(input_image, dtype=tf.float32)
            # output = self.model.build_server_graph(input_image)
            # output = (output + 1.) * 127.5
            # output = tf.reverse(output, [-1])
            # output = tf.saturate_cast(output, tf.uint8)
            # load pretrained model
            self.session.run(self.assign_ops)
            result = self.session.run(self.output, feed_dict={
                    self.input_image: input_image
                })
            #cv2.imwrite(args.output, result[0][:, :, ::-1])
            return result


if __name__ == "__main__":
    ng.get_gpus(1)
    args = parser.parse_args()

    # model = InpaintCAModel()
    image = cv2.imread(args.image)
    mask = cv2.imread(args.mask)

    wrapper = GInpWrapper(args.checkpoint_dir)
    result = wrapper.predict(image, mask)
    cv2.imwrite(args.output, result[0][:, :, ::-1])


    # assert image.shape == mask.shape

    # h, w, _ = image.shape
    # grid = 8
    # image = image[:h//grid*grid, :w//grid*grid, :]
    # mask = mask[:h//grid*grid, :w//grid*grid, :]
    # print('Shape of image: {}'.format(image.shape))

    # image = np.expand_dims(image, 0)
    # mask = np.expand_dims(mask, 0)
    # input_image = np.concatenate([image, mask], axis=2)

    # sess_config = tf.ConfigProto()
    # sess_config.gpu_options.allow_growth = True
    # with tf.Session(config=sess_config) as sess:
    #     print(input_image.shape)
    #     input_image = tf.constant(input_image, dtype=tf.float32)
    #     output = model.build_server_graph(input_image)
    #     output = (output + 1.) * 127.5
    #     output = tf.reverse(output, [-1])
    #     output = tf.saturate_cast(output, tf.uint8)
    #     # load pretrained model
    #     vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    #     assign_ops = []
    #     for var in vars_list:
    #         vname = var.name
    #         from_name = vname
    #         var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
    #         assign_ops.append(tf.assign(var, var_value))
    #     sess.run(assign_ops)
    #     print('Model loaded.')
    #     result = sess.run(output)
    #     cv2.imwrite(args.output, result[0][:, :, ::-1])
