import tensorflow as tf
import scipy
import scipy.stats
import numpy as np
import sys
import os
import pickle
import csv

np.random.seed(46)

iter_time = 1000000
z_dim = 100
batch_size = 64

show_every = 50
gen_every = 400
save_every = 400

img_row = 64
img_col = 64

d_update_time = 1
g_update_time = 1

d_loss_thresh = 1
g_loss_thresh = 1


initial_stddev = 0.02

lr = 2e-4


go_prepro = False
go_train = False
go_test = False

restore_training = False

mode = 'train-prepro'


if mode not in ['train', 'train-prepro', 'test']:
    print('error')
    sys.exit()

if mode == 'train':
    go_prepro = False
    go_train = True
    go_test = False

elif mode == 'train-prepro':
    go_prepro = True
    go_train = True
    go_test = False

elif mode == 'test':
    go_prepro = False
    go_train = False
    go_test = True
    

img_dir = './dataset/faces'
tags_clean_path = './dataset/tags_clean.csv'
test_text_path = sys.argv[1]

prepro_dir = './prepro/'
checkpoint_dir = './hw4_model/'

train_img_dir = './train_img_dir/'
test_output_dir = './samples/'


ckpt_dir1 = './ckpt1/'
ckpt_dir2 = './ckpt2/'
ckpt_dir3 = './ckpt3/'
ckpt_dir4 = './ckpt4/'
ckpt_dir5 = './ckpt5/'


if not os.path.exists(prepro_dir):
    os.makedirs(prepro_dir)
    
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

if not os.path.exists(train_img_dir):
    os.makedirs(train_img_dir)

if not os.path.exists(test_output_dir):
    os.makedirs(test_output_dir)


color_hair_list = [
        'orange hair', 'white hair', 'aqua hair', 'gray hair',
        'green hair', 'red hair', 'purple hair', 'pink hair',
        'blue hair', 'black hair', 'brown hair', 'blonde hair'
        ]


the_color_hair_list = color_hair_list + ['unknown hair']


color_eyes_list = [
        'gray eyes', 'black eyes', 'orange eyes',
        'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
        'green eyes', 'brown eyes', 'red eyes', 'blue eyes'
        ]


the_color_eyes_list = color_eyes_list + ['unknown eyes']


the_attrib_list = the_color_hair_list + the_color_eyes_list
attrib_num = len(the_attrib_list)


if go_prepro:   
    
    img_feats = []
    
    # target_tags = []
    target_tag_hots = []
    
    f = open(tags_clean_path)
    
    for idx, row in enumerate(csv.reader(f)):
        
        tags = row[1]
        
        target_tag = ''
        
        
        valid_target = True
        
        have_color_hair = False
        for color_hair in color_hair_list:
            if color_hair in tags:
                if have_color_hair:
                    valid_target = False
                    break
                have_color_hair = True
                target_tag = target_tag + color_hair
        
        if 'hair' not in target_tag:
            target_tag = target_tag + 'unknown hair'
        
        
        target_tag = target_tag + ' '
        
        have_color_eyes = False
        for color_eyes in color_eyes_list:
            if color_eyes in tags:
                if have_color_eyes:
                    valid_target = False
                    break
                have_color_eyes = True
                target_tag = target_tag + color_eyes
        
        if 'eyes' not in target_tag:
            target_tag = target_tag + 'unknown eyes'
        
        
        
        
        if target_tag == 'unknown hair unknown eyes' or not valid_target:
            continue
        
        
        tag_hot = np.zeros(len(the_attrib_list))
        
        for i, attrib in enumerate(the_attrib_list):
            if attrib in target_tag:
                tag_hot[i] += 1
        
        
        print(idx, target_tag)
        
        img_path = os.path.join(img_dir, str(idx) + '.jpg')
        
        feat1 = scipy.misc.imread(img_path)
        feat1 = scipy.misc.imresize(feat1, [64, 64, 3])
        img_feats.append(feat1)
        # target_tags.append(target_tag)
        target_tag_hots.append(tag_hot)
        
        
        feat2 = np.fliplr(feat1)
        img_feats.append(feat2)
        # target_tags.append(target_tag)
        target_tag_hots.append(tag_hot)
        
        
        feat3 = scipy.misc.imrotate(feat1, 5)
        img_feats.append(feat3)
        # target_tags.append(target_tag)
        target_tag_hots.append(tag_hot)
        
        
        feat4 = scipy.misc.imrotate(feat1, -5)
        img_feats.append(feat4)
        # target_tags.append(target_tag)
        target_tag_hots.append(tag_hot)
        
    f.close()
    
    
    img_feats = np.asarray(img_feats, dtype = 'float32')/127.5 - 1.
    target_tag_hots = np.asarray(target_tag_hots)
    
    
    
    img_feats_1 = img_feats[:int(len(img_feats)/2), :]
    img_feats_2 = img_feats[int(len(img_feats)/2):, :]
    
    
    # with open(os.path.join(prepro_dir, 'img_feats'), 'wb') as fp:
    #    pickle.dump(img_feats, fp)
    
    with open(os.path.join(prepro_dir, 'img_feats_1'), 'wb') as fp:
        pickle.dump(img_feats_1, fp)
    
    with open(os.path.join(prepro_dir, 'img_feats_2'), 'wb') as fp:
        pickle.dump(img_feats_2, fp)
    
    
        
    with open(os.path.join(prepro_dir, 'target_tag_hots'), 'wb') as fp:
        pickle.dump(target_tag_hots, fp)




# test_target_tags = []
test_target_tag_hots = []

f = open(test_text_path, 'r')

for idx, line in enumerate(f.readlines()):
    
    tags = line.split(',')[1]
    
    target_tag = ''
    
    
    valid_target = True
    
    have_color_hair = False
    for color_hair in color_hair_list:
        if color_hair in tags:
            if have_color_hair:
                valid_target = False
                break
            have_color_hair = True
            target_tag = target_tag + color_hair
    
    if 'hair' not in target_tag:
        target_tag = target_tag + 'unknown hair'
    
    
    target_tag = target_tag + ' '
    
    have_color_eyes = False
    for color_eyes in color_eyes_list:
        if color_eyes in tags:
            if have_color_eyes:
                valid_target = False
                break
            have_color_eyes = True
            target_tag = target_tag + color_eyes
    
    if 'eyes' not in target_tag:
        target_tag = target_tag + 'unknown eyes'
    
    
    
    
    if target_tag == 'unknown hair unknown eyes' or not valid_target:
        continue
    
    
    tag_hot = np.zeros(len(the_attrib_list))
    
    for i, attrib in enumerate(the_attrib_list):
        if attrib in target_tag:
            tag_hot[i] += 1
    
    
    print(idx, target_tag)
    
    # test_target_tags.append(target_tag)
    test_target_tag_hots.append(tag_hot)
    
f.close()

test_target_tag_hots = np.asarray(test_target_tag_hots)




class Data(object):
    def __init__(self, img_feats, tag_hots, z_dim):
        
        self.img_feats = img_feats
        self.tag_hots = tag_hots
        self.z_dim = z_dim
        self.size = img_feats.shape[0]
    
    def next_batch(self, batch_size):
        
        selected_idx = np.random.permutation(self.size)[:batch_size]
        
        self.batch_img_feats = self.img_feats[selected_idx,:]
        self.batch_tag_hots = self.tag_hots[selected_idx,:]
        
        wrong_idx1 = np.random.permutation(self.size)[:batch_size]
        wrong_idx2 = np.random.permutation(self.size)[:batch_size]
        
        self.wrong_batch_img_feats = self.img_feats[wrong_idx1,:]
        self.wrong_tag_hots = self.tag_hots[wrong_idx2,:]
        
        
        
        return self.batch_img_feats, self.batch_tag_hots, self.wrong_batch_img_feats, self.wrong_tag_hots


def step_dump_img(train_img_dir, img_feats, steps):
    
    img_feats = (img_feats + 1) * 127.5
    img_feats = np.asarray(img_feats, dtype = np.uint8)
    
    output_img_feats = []
    for i in range(len(img_feats)):
        output_img_feats.append(scipy.misc.imresize(img_feats[i], [64, 64, 3]))
    output_img_feats = np.asarray(output_img_feats)
    
    for idx, img_feat in enumerate(output_img_feats):
        path = os.path.join(train_img_dir, 'step_' + str(steps) + '_sample_'+ str(idx+1) + '.jpg')
        scipy.misc.imsave(path, img_feat)
        
        
def dump_img(test_output_dir, img_feats, sample_num):
        
    img_feats = (img_feats + 1) * 127.5
    img_feats = np.asarray(img_feats, dtype = np.uint8)
    
    output_img_feats = []
    for i in range(len(img_feats)):
        output_img_feats.append(scipy.misc.imresize(img_feats[i], [64, 64, 3]))
    output_img_feats = np.asarray(output_img_feats)
    
    
    for idx, img_feat in enumerate(output_img_feats):
        path = os.path.join(test_output_dir, 'sample_' + str(idx + 1) + '_' + str(sample_num) + '.jpg')
        scipy.misc.imsave(path, img_feat)


def leaky_relu(x, alpha = 0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


class Generator(object):
    def __init__(self, img_row, img_col):
        
        self.img_row = img_row
        self.img_col = img_col
        
    
    def __call__(self, tags, z, reuse = False, train = True):
        
        with tf.variable_scope("g_net") as scope:
            if reuse:
                scope.reuse_variables()
                
            ztags_vec = tf.concat([tags, z], axis = 1)
            
            
            fc = tf.contrib.layers.fully_connected(
                    ztags_vec, 4 * 4 * 256,
                    weights_initializer = tf.random_normal_initializer(stddev = initial_stddev),
                    activation_fn = None
                    )
            
            
            fc = tf.layers.batch_normalization(fc, training = train)
            fc = tf.reshape(fc, [-1, 4, 4, 256])
            fc = tf.nn.relu(fc)
            
            
            conv1 = tf.contrib.layers.conv2d_transpose(
                    fc,
                    128, 6, 2,
                    weights_initializer = tf.random_normal_initializer(stddev = initial_stddev),
                    activation_fn = None
                    )
            
            
            conv1 = tf.layers.batch_normalization(conv1, training = train)
            conv1 = tf.nn.relu(conv1)
            
            
            conv2 = tf.contrib.layers.conv2d_transpose(
                    conv1,
                    64, 5, 2,
                    weights_initializer = tf.random_normal_initializer(stddev = initial_stddev),
                    activation_fn = None
                    )
            
            
            conv2 = tf.layers.batch_normalization(conv2, training = train)
            conv2 = tf.nn.relu(conv2)
            
            
            conv3 = tf.contrib.layers.conv2d_transpose(
                    conv2,
                    32, 4, 2,
                    weights_initializer = tf.random_normal_initializer(stddev = initial_stddev),
                    activation_fn = None
                    )
            
            
            conv3 = tf.layers.batch_normalization(conv3, training = train)
            conv3 = tf.nn.relu(conv3)
            
            
            conv4 = tf.contrib.layers.conv2d_transpose(
                    conv3,
                    3, 4, 2,
                    weights_initializer = tf.random_normal_initializer(stddev = initial_stddev),
                    activation_fn = None
                    )
            conv4 = tf.nn.tanh(conv4)
            return conv4
    
    
    @property
    def vars(self):
        return [var for var in tf.global_variables() if "g_net" in var.name]




class Discriminator(object):
    def __init__(self, img_row, img_col):
        
        self.img_row = img_row
        self.img_col = img_col
        
    
    def __call__(self, tags, imgs, reuse = True):
        
        with tf.variable_scope("d_net") as scope:
            
            if reuse == True:
                scope.reuse_variables()
            
            
            # deconv
            
            deconv1 = tf.contrib.layers.conv2d(
                    imgs,
                    32, 4, 2,
                    weights_initializer = tf.random_normal_initializer(stddev = initial_stddev),
                    activation_fn = None
                    )
            deconv1 = tf.layers.batch_normalization(deconv1, training = True)
            deconv1 = leaky_relu(deconv1)
            
            
            deconv2 = tf.contrib.layers.conv2d(
                    deconv1,
                    64, 5, 2,
                    weights_initializer = tf.random_normal_initializer(stddev = initial_stddev),
                    activation_fn = None
                    )
            deconv2 = tf.layers.batch_normalization(deconv2, training = True)
            deconv2 = leaky_relu(deconv2)
            
            
            deconv3 = tf.contrib.layers.conv2d(
                    deconv2,
                    128, 6, 2,
                    weights_initializer = tf.random_normal_initializer(stddev = initial_stddev),
                    activation_fn = None
                    )
            deconv3 = tf.layers.batch_normalization(deconv3, training = True)
            deconv3 = leaky_relu(deconv3)
            
            
            tags_vec = tf.expand_dims(tf.expand_dims(tags, 1), 2)
            tags_vec = tf.tile(tags_vec, [1, 8, 8, 1])
            
            
            pair = tf.concat([deconv3, tags_vec], axis = -1)
            
            conv4 = tf.contrib.layers.conv2d(
                    pair,
                    128, 1, 1,
                    weights_initializer = tf.random_normal_initializer(stddev = initial_stddev),
                    activation_fn = None
                    )
            conv4 = tf.layers.batch_normalization(conv4, training = True)
            conv4 = leaky_relu(conv4)
            
            conv5 = tf.contrib.layers.conv2d(
                    conv4,
                    1, 8, 1,
                    weights_initializer = tf.random_normal_initializer(stddev = initial_stddev),
                    padding = 'VALID',
                    activation_fn = None
                    )
            
            output = tf.squeeze(conv5, [1, 2, 3])
            
            return output
    
    @property
    def vars(self):
        return [var for var in tf.global_variables() if "d_net" in var.name]




class GAN(object):
    def __init__(self):
        
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        
        self.sess = tf.Session(config = config)
        
        self.attrib_num = attrib_num
        
        self.img_row = img_row
        self.img_col = img_col
        
        self.z_dim = z_dim
        
        self.test_target_tag_hots = test_target_tag_hots
        
        self.checkpoint_dir = checkpoint_dir
    
    def build_model(self):
        
        
        self.g_net = Generator(img_row = img_row, img_col = img_col)
        self.d_net = Discriminator(img_row = img_row, img_col = img_col)
        
        
        self.tags = tf.placeholder(tf.float32, [None, self.attrib_num])
        self.imgs = tf.placeholder(tf.float32, [None, self.img_row, self.img_col, 3])
        
        self.z = tf.placeholder(tf.float32, [None, self.z_dim])
        
        self.wrong_tags = tf.placeholder(tf.float32, [None, self.attrib_num])
        self.wrong_imgs = tf.placeholder(tf.float32, [None, self.img_row, self.img_col, 3])
        
        self.real_tags = self.tags
        self.real_imgs = self.imgs
        
        self.fake_imgs = self.g_net(self.real_tags, self.z)
        
        self.sampler = tf.identity(self.g_net(self.real_tags, self.z, reuse = True, train = False)) 
        
        self.d_logits_0 = self.d_net(self.real_tags, self.real_imgs, reuse = False)
        self.d_logits_1 = self.d_net(self.real_tags, self.fake_imgs)
        self.d_logits_2 = self.d_net(self.wrong_tags, self.real_imgs)
        self.d_logits_3 = self.d_net(self.real_tags, self.wrong_imgs)
        
        
        
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_logits_1, labels = tf.ones_like(self.d_logits_1)))
        
        self.d_loss = 0
        self.d_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_logits_1, labels = tf.zeros_like(self.d_logits_1)))
        self.d_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_logits_2, labels = tf.zeros_like(self.d_logits_2)))
        self.d_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_logits_3, labels = tf.zeros_like(self.d_logits_3)))
        self.d_loss /= 3
        self.d_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_logits_0, labels=tf.ones_like(self.d_logits_0)))
        
        
        self.global_step = tf.Variable(0, trainable = False)
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_updates = tf.train.AdamOptimizer(lr).minimize(self.d_loss, var_list = self.d_net.vars)
            self.g_updates = tf.train.AdamOptimizer(lr).minimize(self.g_loss, var_list = self.g_net.vars, global_step = self.global_step)
            
        
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(tf.global_variables())
        
    
    def load_ckpt(self, ckpt_dir):
        print('ckpt dir:', ckpt_dir)
        self.saver = tf.train.Saver()
        self.ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if self.ckpt and self.ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
        self.sess.run(tf.global_variables())
        
    
    def train(self, data):
        
        self.data = data
        
        for t in range(iter_time):
            
            img, tags, w_img, w_tags = self.data.next_batch(batch_size)
            
            for d_upd in range(d_update_time):
                
                # img, tags, w_img, w_tags = self.data.next_batch(batch_size)
                
                z = np.random.sample((batch_size, self.z_dim)) * 2 - 1
                
                feed_dict = {
                        self.tags:tags,
                        self.imgs:img,
                        self.z:z,
                        self.wrong_tags:w_tags,
                        self.wrong_imgs:w_img
                        }
                
                _, loss = self.sess.run([self.d_updates, self.d_loss], feed_dict = feed_dict)
                the_d_loss = loss
                # print('the d loss', the_d_loss)
                
                if the_d_loss < d_loss_thresh:
                    break
            
            for g_upd in range(g_update_time):
                
                # img, tags, w_img, w_tags = self.data.next_batch(batch_size)
                
                z = np.random.sample((batch_size, self.z_dim)) * 2 - 1
                
                feed_dict = {
                        self.tags:tags,
                        self.imgs:img,
                        self.z:z,
                        self.wrong_tags:w_tags,
                        self.wrong_imgs:w_img
                        }
                
                _, loss, step = self.sess.run([self.g_updates, self.g_loss, self.global_step], feed_dict = feed_dict)
                the_g_loss = loss
                # print('the g loss', the_g_loss)
                
                if the_g_loss < g_loss_thresh:
                    break
            
            
            current_step = tf.train.global_step(self.sess, self.global_step)
            
            if current_step % show_every == 0:
                
                print("\n")
                print("Step:", current_step)
                print("Discriminator loss:", the_d_loss)
                print("Generator loss:", the_g_loss)
                print("\n")
                
            if current_step % save_every == 0:
                path = self.saver.save(self.sess, self.checkpoint_dir, global_step = current_step)
                print ("Save ckpt", path)
            
            
            if current_step % gen_every == 0:
                print("Testing Go!")
                self.gen_img(self.test_target_tag_hots)
                self.step_gen_img(current_step)
                print("Testing Done!")
                
                
    
    def test(self, test_target_tag_hots, sample_number):
        print("Testing Go!")
        self.test_gen_img(test_target_tag_hots, sample_number)
        print("Testing Done!")
        
    
    
    def gen_img(self, test_target_tag_hots):
        
        for i in range(5):
            
            tags = test_target_tag_hots
            
            test_batch_size = len(tags)
            z = np.random.sample((test_batch_size, self.z_dim))  * 2 - 1
                
            
            feed_dict = {
                    self.tags:tags,
                    self.z:z
                    }
            
            fake_imgs = self.sess.run(self.sampler, feed_dict = feed_dict)
            dump_img(test_output_dir, fake_imgs, i + 1)
    
    
    def test_gen_img(self, test_target_tag_hotsm, sample_number):
        
        tags = test_target_tag_hots
        
        test_batch_size = len(tags)
        z = np.random.sample((test_batch_size, self.z_dim))  * 2 - 1
                
        feed_dict = {
                self.tags:tags,
                self.z:z
                }
        
        fake_imgs = self.sess.run(self.sampler, feed_dict = feed_dict)
        dump_img(test_output_dir, fake_imgs, sample_number)
            
    
    def step_gen_img(self, steps):
        
        tags = self.test_target_tag_hots
        
        test_batch_size = len(tags)
        z = np.random.sample((test_batch_size, self.z_dim))  * 2 - 1
        
        feed_dict = {
                self.tags:tags,
                self.z:z
                }
        
        fake_imgs = self.sess.run(self.sampler, feed_dict = feed_dict)
        step_dump_img(train_img_dir, fake_imgs, steps)
        
        
    
if go_train:
    
    # with open(os.path.join(prepro_dir, 'img_feats'), 'rb') as fp:
    #    img_feats = pickle.load(fp)
            
    with open(os.path.join(prepro_dir, 'img_feats_1'), 'rb') as fp:
        img_feats_1 = pickle.load(fp)
    
    with open(os.path.join(prepro_dir, 'img_feats_2'), 'rb') as fp:
        img_feats_2 = pickle.load(fp)
        
    img_feats = np.concatenate((img_feats_1, img_feats_2), axis = 0)
    
        
    with open(os.path.join(prepro_dir, 'target_tag_hots'), 'rb') as fp:
        target_tag_hots = pickle.load(fp)
    
    
    print('Training Data Size:', img_feats.shape)
    
    data = Data(img_feats, target_tag_hots, z_dim)
    
    model = GAN()
    model.build_model()

    if restore_training:
        model.load_ckpt(checkpoint_dir)
    
    model.train(data)
    
elif go_test:
    
    model = GAN()
    model.build_model()
    
    model.load_ckpt(ckpt_dir1)
    model.test(test_target_tag_hots, 1)
    
    model.load_ckpt(ckpt_dir2)
    model.test(test_target_tag_hots, 2)
    
    model.load_ckpt(ckpt_dir3)
    model.test(test_target_tag_hots, 3)
    
    model.load_ckpt(ckpt_dir4)
    model.test(test_target_tag_hots, 4)
    
    model.load_ckpt(ckpt_dir5)
    model.test(test_target_tag_hots, 5)