import os
import numpy as np
from imageio import imread, imwrite
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import tensorflow as tf
from collections import OrderedDict

from image_utils import combine_img, threshold_mean, mirror_combine_data, reverse_flow_combine_data, generate_set_from_idx
from tf_utils import weight_variable, bias_variable, conv2d, crop
from io_utils import read_masks, read_inputs

def conv2d_same(x, W, b):
    conv = tf.nn.conv2d(x, W, strides=[ 1, 1, 1, 1 ], padding='SAME')
    return conv + b

def batch_activ_conv(x, is_training, keep_prob, W, b):
    x = tf.contrib.layers.batch_norm(x, scale=True, is_training=is_training, updates_collections=None)
    x = tf.nn.relu(x)
    x = conv2d_same(x, W, b)
    x = tf.nn.dropout(x, keep_prob)
    return x

def block_flow(x, layers, in_features, growth, is_training, keep_prob, weights):
    current = x
    features = in_features
    for idx in range(layers):
        print(current.get_shape())
        W = weight_variable([ 3, 3, features, growth ])
        weights.append(W)
        b = bias_variable([ growth ])
        tmp = batch_activ_conv(current, is_training, keep_prob, W, b)
        current = tf.concat((current, tmp), axis=3)
        features += growth
    return current, features, weights

def block_single(x1, x2, x3, layers, in_features, growth, is_training, keep_prob, weights):
    current1 = x1
    current2 = x2
    current3 = x3
    features = in_features
    for idx in range(layers):
        print(current1.get_shape())
        W = weight_variable([ 3, 3, features, growth ])
        weights.append(W)
        b = bias_variable([ growth ])
        tmp1 = batch_activ_conv(current1, is_training, keep_prob, W, b)
        tmp2 = batch_activ_conv(current2, is_training, keep_prob, W, b)
        tmp3 = batch_activ_conv(current3, is_training, keep_prob, W, b)
        current1 = tf.concat((current1, tmp1), axis=3)
        current2 = tf.concat((current2, tmp2), axis=3)
        current3 = tf.concat((current3, tmp3), axis=3)
        features += growth
    return current1, current2, current3, features, weights

def avg_pool(x, s):
    return tf.nn.avg_pool(x, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID')

if __name__ == '__main__':
    #print(os.listdir("Trainingsdata"))
    plt.close("all")
    inputs = []
    previouss = []
    nexts = []
    
    masks = []
    
    
    for f in sorted(os.listdir("Trainingsdata")):
        input_string = f + "/roi/input.tif" 
        previous_string = f + "/roi/previous.tif" 
        next_string = f + "/roi/next.tif" 
        mask_string = f + "/roi/mask.tif" 
        inputs.append(os.path.join("Trainingsdata", input_string))
        previouss.append(os.path.join("Trainingsdata", previous_string))
        nexts.append(os.path.join("Trainingsdata", next_string))
        masks.append(os.path.join("Trainingsdata", mask_string))
    '''
    img = imread(inputs[2])
    gray = color.rgb2gray(img) 
    print(gray.shape)
    
    #img = np.reshape(img, ((1,) + img.shape))

    
    mask = imread(masks[2])
    print(mask.shape)
    plt.figure()
    plt.imshow(img)
    plt.figure()
    plt.imshow(gray, cmap="gray")
    plt.figure()
    plt.imshow(mask, cmap="gray")
    '''
    #80% training, 20% test data
    num_items = len(masks)
    ind_train = np.int64(num_items * 0.8)
    
    masks_train_list = masks[:ind_train]
    masks_test_list = masks[ind_train:]

    inputs_train_list = inputs[:ind_train]
    inputs_test_list = inputs[ind_train:]
    
    previouss_train_list = previouss[:ind_train]
    previouss_test_list = previouss[ind_train:]
    
    nexts_train_list = nexts[:ind_train]
    nexts_test_list = nexts[ind_train:]
    
    #roi = 64
    roi = 64
    size = np.int(512 / roi)
    
    padding_width = 0
    
    masks_train = np.zeros((len(masks_train_list)*size*size, roi+2*padding_width, roi+2*padding_width))
    inputs_train = np.zeros((len(inputs_train_list)*size*size, roi+2*padding_width, roi+2*padding_width, 3))
    previous_train = np.zeros((len(inputs_train_list)*size*size, roi+2*padding_width, roi+2*padding_width, 3))
    next_train = np.zeros((len(inputs_train_list)*size*size, roi+2*padding_width, roi+2*padding_width, 3))

    masks_test = np.zeros((len(masks_test_list)*size*size, roi+2*padding_width, roi+2*padding_width))
    inputs_test = np.zeros((len(inputs_test_list)*size*size, roi+2*padding_width, roi+2*padding_width, 3))
    previous_test = np.zeros((len(inputs_test_list)*size*size, roi+2*padding_width, roi+2*padding_width, 3))
    next_test = np.zeros((len(inputs_test_list)*size*size, roi+2*padding_width, roi+2*padding_width, 3))
    
    
    masks_train = read_masks(masks_train_list, size, roi, padding_width)
    inputs_train = read_inputs(inputs_train_list, size, roi, padding_width)
    previous_train = read_inputs(previouss_train_list, size, roi, padding_width)
    next_train = read_inputs(nexts_train_list, size, roi, padding_width)
    
    
    masks_test = read_masks(masks_test_list, size, roi, padding_width)
    inputs_test = read_inputs(inputs_test_list, size, roi, padding_width)
    previous_test = read_inputs(previouss_test_list, size, roi, padding_width)
    next_test = read_inputs(nexts_test_list, size, roi, padding_width)
    
   # sort out black masks for training
    
    full_masks_train = masks_train
    full_masks_train = mirror_combine_data(full_masks_train)
    full_masks_train = np.vstack((full_masks_train, full_masks_train))
    ind = threshold_mean(full_masks_train, 0.01, 0.8)
    masks_train = full_masks_train[ind]
    
    #masks_train = (~masks_train.astype(bool)).astype(int)
    #full_masks_train = (~full_masks_train.astype(bool)).astype(int)
    
    #masks_train = masks_train[0]
    
    #masks_train = np.reshape(masks_train, (1, masks_train.shape[0], masks_train.shape[1]))
    
    full_training_set = np.zeros((inputs_train.shape[0], inputs_train.shape[1], inputs_train.shape[2], 9))
    full_training_set[:,:,:,0:3] = previous_train
    full_training_set[:,:,:,3:6] = inputs_train
    full_training_set[:,:,:,6:9] = next_train
    
    full_training_set = mirror_combine_data(full_training_set)
    full_training_set = reverse_flow_combine_data(full_training_set)
    
    training_set = full_training_set[ind]
    #training_set = np.reshape(training_set, (1, training_set.shape[0], training_set.shape[1], training_set.shape[2]))
    

    ind = threshold_mean(masks_test, 0.01, 0.8)
    #masks_test = (~masks_test.astype(bool)).astype(int)
    full_masks_test = masks_test
    #full_masks_test = (~full_masks_test.astype(bool)).astype(int)
    masks_test = masks_test[ind]
    
    
    full_test_set = np.zeros((inputs_test.shape[0], inputs_test.shape[1], inputs_test.shape[2], 9))
    full_test_set[:,:,:,0:3] = previous_test
    full_test_set[:,:,:,3:6] = inputs_test
    full_test_set[:,:,:,6:9] = next_test
    
    test_set = full_test_set[ind]
    #full_test_set = full_test_set[ind]
    
    print(full_training_set.shape, training_set.shape)
    print(full_test_set.shape, test_set.shape)
        
    
    test_set_large_area = generate_set_from_idx(full_test_set, np.array([1,2,3,12,13,14,16,19]), size)
    test_set_small_area = generate_set_from_idx(full_test_set, np.array([4,6,7,8,9,10,11,15,17,18]), size)
    test_set_large_flow = generate_set_from_idx(full_test_set,np.array([2,3,5,11,13,16,17,18]), size)
    test_set_small_flow = generate_set_from_idx(full_test_set,np.array([1,4,6,7,8,10,15,19]), size)
    
    masks_test_large_area = generate_set_from_idx(full_masks_test, np.array([1,2,3,12,13,14,16,19]), size)
    masks_test_small_area = generate_set_from_idx(full_masks_test, np.array([4,6,7,8,9,10,11,15,17,18]), size)
    masks_test_large_flow = generate_set_from_idx(full_masks_test,np.array([2,3,5,11,13,16,17,18]), size)
    masks_test_small_flow = generate_set_from_idx(full_masks_test,np.array([1,4,6,7,8,10,15,19]), size)
    
    
    #print(ind)
    #asdf

    #tensorflow
    
    n_channels = 9
    layers = 2
    features = 40
    filter_size = 3
    keep_prob = 0.85
    layers_flow = 5
    layers = 3
    
    s = tf.placeholder(tf.float32, shape = [None, masks_train.shape[1], masks_train.shape[2]])
    u = tf.placeholder(tf.float32, shape = [None, inputs_train.shape[1], inputs_train.shape[1], n_channels])
    is_training = tf.placeholder(tf.bool)

    s_img = tf.reshape(s, [-1, masks_train.shape[1], masks_train.shape[2], 1])
    u_img = tf.reshape(u, [-1, inputs_train.shape[1], inputs_train.shape[2], n_channels])
    
    in_node = u_img
    in_node_test = u_img
    
    in_node1 = u_img[:,:,:,0:3]
    in_node2 = u_img[:,:,:,3:6]
    in_node3 = u_img[:,:,:,6:9]
    
    in_node1_test = u_img[:,:,:,0:3]
    in_node2_test = u_img[:,:,:,3:6]
    in_node3_test = u_img[:,:,:,6:9]
 

    
    weights = []
    biases = []
    convs = []
    dw_h_convs = OrderedDict()
    
    convs1 = []
    dw_h_convs_test = OrderedDict()
    dw_h_convs1 = OrderedDict()
    dw_h_convs1_test = OrderedDict()
    convs2 = []
    dw_h_convs2 = OrderedDict()
    dw_h_convs2_test = OrderedDict()
    convs3 = []
    dw_h_convs3 = OrderedDict()
    dw_h_convs3_test = OrderedDict()
    
    convs_comb = []
    dw_h_convs_comb = OrderedDict()
    dw_h_convs_comb_test = OrderedDict()
    
    
    #flow path
    W = weight_variable([filter_size, filter_size, n_channels, 16])
    b = bias_variable([16])
    weights.append(W)
    
    current_flow = conv2d_same(in_node, W, b)
    current_flow, features, weights = block_flow(current_flow, layers_flow, 16, 12, is_training, keep_prob, weights)    
    print("Flow after Block1:", current_flow.get_shape())
    
    W = weight_variable([1, 1, features, features//4])
    b = bias_variable([features//4])
    weights.append(W)
    
    current_flow = batch_activ_conv(current_flow, is_training, keep_prob, W, b)
    #current_flow = avg_pool(current_flow, 2)
    current_flow, features, weights = block_flow(current_flow, layers_flow, features//4, 12, is_training, keep_prob, weights)    
    print("Flow after Block2:", current_flow.get_shape())
    
    W = weight_variable([1, 1, features, features])
    b = bias_variable([features])
    weights.append(W)
    
    current_flow = batch_activ_conv(current_flow, is_training, keep_prob, W, b)
    
    
    #single path
    W = weight_variable([filter_size, filter_size, n_channels//3, 16])
    b = bias_variable([16])
    weights.append(W)
    
    current_1 = conv2d_same(in_node1, W, b)
    current_2 = conv2d_same(in_node2, W, b)
    current_3 = conv2d_same(in_node3, W, b)
    #print(current_1.get_shape())
    current_1, current_2, current_3, features, weights = block_single(current_1, current_2, current_3, layers, 16, 12, is_training, keep_prob, weights)    
    print("Single after Block1:", current_1.get_shape())
    
    W = weight_variable([1, 1, features, features//4])
    b = bias_variable([features//4])
    weights.append(W)
    
    current_1 = batch_activ_conv(current_1, is_training, keep_prob, W, b)
    current_2 = batch_activ_conv(current_2, is_training, keep_prob, W, b)
    current_3 = batch_activ_conv(current_3, is_training, keep_prob, W, b)
    
    #current_1 = avg_pool(current_1, 2)
    #current_2 = avg_pool(current_2, 2)
    #current_3 = avg_pool(current_3, 2)
    
    current_1, current_2, current_3, features, weights = block_single(current_1, current_2, current_3, layers, features//4, 12, is_training, keep_prob, weights)    
    print("Single after Block2:", current_1.get_shape())
        
    W = weight_variable([1, 1, features, features])
    b = bias_variable([features])
    weights.append(W)
    
    current_1 = batch_activ_conv(current_1, is_training, keep_prob, W, b)
    current_2 = batch_activ_conv(current_2, is_training, keep_prob, W, b)
    current_3 = batch_activ_conv(current_3, is_training, keep_prob, W, b)
    
    
    #combine

    current = tf.concat([current_flow, current_1, current_2, current_3], 3)
    print("after concat:", current.get_shape())
    
    W = weight_variable([filter_size, filter_size, current.get_shape()[3].value, 40])
    b = bias_variable([40])
    weights.append(W)
    
    current = conv2d_same(current, W, b)
    
    W = weight_variable([1, 1, 40, 40])
    b = bias_variable([40])
    weights.append(W)
    
    current = batch_activ_conv(current, is_training, keep_prob, W, b)
    
    weight = weight_variable([1, 1, 40, 1])
    bias = bias_variable([1],-0.5)
    conv = conv2d_same(current, weight, tf.constant(1.0))
    conv_test = conv
    pred_mask = tf.nn.sigmoid(conv + bias)
    pred_mask_test = pred_mask
    
    
    print(pred_mask.get_shape())
    
    
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    # loss

    #loss = tf.losses.mean_squared_error(pred_mask, crop(s_img,pred_mask))
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=s_, labels=s))  
    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=s_, labels=s))     
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=conv+bias, labels=crop(s_img,pred_mask))) #sigmoid
    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_mask, labels=crop(s_img,pred_mask))) #relu
    loss_test = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=conv_test+bias, labels=crop(s_img,pred_mask_test))) #sigmoid
    
    # l2 regularization = weight decay
    beta = 0.01
    regularizers = 0
    for reg in weights:
        regularizers += tf.nn.l2_loss(reg)
            
    loss = tf.reduce_mean(loss + beta * regularizers)
    loss_test = tf.reduce_mean(loss_test + beta * regularizers)
    
    
    loss_sum = tf.summary.scalar("cross_entropy_densenet_comb", loss)
    
    
    loss_batch = tf.placeholder(tf.float32)
    loss_batch_summary = tf.summary.scalar("cross_entropy_densenet_comb", loss_batch)
    
    
    #accuracy
    
    #correct_prediction = tf.equal(tf.round(tf.clip_by_value(pred_mask, 0., 1.),),crop(s_img,pred_mask)) # relu
    correct_prediction = tf.equal(tf.round(pred_mask,),crop(s_img,pred_mask)) # sigmoid
    correct_prediction_test = tf.equal(tf.round(pred_mask_test,),crop(s_img,pred_mask_test)) # sigmoid
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
    accuracy_test = tf.reduce_mean(tf.cast(correct_prediction_test, tf.float64))
    
    accuracy_sum = tf.summary.scalar("accuracy_densenet_comb", accuracy)
    
    accuracy_batch = tf.placeholder(tf.float32)
    accuracy_batch_summary = tf.summary.scalar("accuracy_densenet_comb", accuracy_batch)
    
    
    merged = tf.summary.merge([loss_sum, accuracy_sum])
    merged_batch = tf.summary.merge([loss_batch_summary, accuracy_batch_summary])
    
    
    train_writer = tf.summary.FileWriter(os.getcwd() + '/tb/train')
    test_writer = tf.summary.FileWriter(os.getcwd() + '/tb/test')
    test_large_area_writer = tf.summary.FileWriter(os.getcwd() + '/tb/test_LA')
    test_small_area_writer = tf.summary.FileWriter(os.getcwd() + '/tb/test_SA')
    test_large_flow_writer = tf.summary.FileWriter(os.getcwd() + '/tb/test_LF')
    test_small_flow_writer = tf.summary.FileWriter(os.getcwd() + '/tb/test_SF')
    
    
    
    #model training
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    #train_step = tf.train.RMSPropOptimizer(0.001).minimize(loss)
    
    init = tf.global_variables_initializer()
    #sess = tf.Session()
     
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(init)
    #train_writer = tf.summary.FileWriter( './logs/1/train ', sess.graph)
    
    Nep = 1000
    
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    batch_size = 100
    n_batches = int(np.ceil(training_set.shape[0] / batch_size))
    
    inputs_train_batch = np.array_split(training_set, n_batches, axis = 0)
    masks_train_batch = np.array_split(masks_train, n_batches, axis = 0)
    
    
    saver = tf.train.Saver()
    current_directory = os.getcwd()
    model_directory = os.path.join(current_directory, r'densenet_comb')
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    
    for i in range(0, Nep):
        if i % 50 == 0:
            print(i)
            
        for inputs_train_minibatch, masks_train_minibatch in zip(inputs_train_batch, masks_train_batch):    
            sess.run([train_step, extra_update_ops], feed_dict={u: inputs_train_minibatch, s: masks_train_minibatch, is_training:True})
                     
        #sess.run(train_step, feed_dict={u: training_set, s: masks_train})
        
        #summary, train_loss = sess.run([merge, loss], feed_dict={u: inputs_train, s: masks_train, keep_prob: 1.0})
        train_loss = 0
        train_acc = 0
        for inputs_train_minibatch, masks_train_minibatch in zip(inputs_train_batch, masks_train_batch):    
            #sess.run(train_step, feed_dict={u: inputs_train_minibatch, s: masks_train_minibatch})
            train_loss_temp, train_acc_temp = sess.run([loss, accuracy], feed_dict={u: inputs_train_minibatch, s: masks_train_minibatch, is_training:False})
            
            train_loss += train_loss_temp
            train_acc += train_acc_temp
        
        train_summary = sess.run(merged_batch, feed_dict={loss_batch: train_loss/n_batches, accuracy_batch: train_acc/n_batches})
            
        train_losses.append(train_loss/n_batches)
        train_accs.append(train_acc/n_batches)

        
        test_loss, test_acc, test_summary = sess.run([loss_test, accuracy_test, merged], feed_dict={u: test_set, s: masks_test, is_training:False})
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        if i == 0: 
            print(test_loss, i)
            best_loss = test_loss
            best_iter = i
            save_path = saver.save(sess, model_directory+"/model")
            print("Model saved in path: %s" % save_path)

        if test_loss < best_loss:
            print(test_loss, i)
            best_loss = test_loss
            best_iter = i
            save_path = saver.save(sess, model_directory+"/model")
            print("Model saved in path: %s" % save_path)
        
        train_writer.add_summary(train_summary, i)
        test_writer.add_summary(test_summary, i)
        
        _, _, test_LA_summary = sess.run([loss_test, accuracy_test, merged], feed_dict={u: test_set_large_area, s: masks_test_large_area, is_training:False})
        _, _, test_SA_summary = sess.run([loss_test, accuracy_test, merged], feed_dict={u: test_set_small_area, s: masks_test_small_area, is_training:False})
        _, _, test_LF_summary = sess.run([loss_test, accuracy_test, merged], feed_dict={u: test_set_large_flow, s: masks_test_large_flow, is_training:False})
        _, _, test_SF_summary = sess.run([loss_test, accuracy_test, merged], feed_dict={u: test_set_small_flow, s: masks_test_small_flow, is_training:False})
        

        test_large_area_writer.add_summary(test_LA_summary, i)
        test_small_area_writer.add_summary(test_SA_summary, i)
        test_large_flow_writer.add_summary(test_LF_summary, i)
        test_small_flow_writer.add_summary(test_SF_summary, i)
        
        print(train_loss/n_batches, test_loss, train_acc/n_batches, test_acc)
      
    plt.figure()
    plt.plot(train_losses, lw = 2, label='train loss')
    plt.plot(test_losses, lw = 2, label='test loss')
    plt.legend()
    plt.savefig('loss.png')
    #plt.show()
    
    plt.figure()
    plt.plot(train_accs, lw = 2, label='train accuracy')
    plt.plot(test_accs, lw = 2, label='test accuracy')
    plt.legend()
    plt.savefig('accuracy.png')
    #plt.show()
    
    
    saver.restore(sess, save_path)
    test_loss, _, _ = sess.run([loss_test, accuracy_test, merged], feed_dict={u: test_set, s: masks_test, is_training:False})
    print(test_loss)
    
    
    n = 15
    n_patches = n * (512//roi)**2 
    
    train_mask = sess.run(pred_mask_test, feed_dict={u: full_training_set[0:n_patches], s: masks_train, is_training:False})
    print(train_mask.shape)
    
    combined_masks = np.zeros((n, 512,512))
    combined_GT_masks = np.zeros((n, 512,512))
    combined_input = np.zeros((n, 512,512,3))
    
    for i in range(n):
        combined_masks[i] = combine_img(train_mask[i*size*size:(i+1)*size*size,:,:,0], size, roi)
        combined_GT_masks[i] = combine_img(full_masks_train[size*size*(i):size*size*(i+1),:,:], size, roi)
        combined_input[i] = combine_img(full_training_set[size*size*(i):size*size*(i+1),:,:,3:6], size, roi)

    thresh_masks = np.ones_like(combined_masks)
    
    ind = np.where(combined_masks < 0.5)
    thresh_masks[ind] = 0.
    '''
    for i in range(n):
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(combined_GT_masks[i], cmap="gray")
        plt.subplot(2,2,2)
        plt.imshow(combined_masks[i], cmap="gray")
        plt.subplot(2,2,3)
        plt.imshow(thresh_masks[i], cmap="gray")
        plt.subplot(2,2,4)
        plt.imshow(combined_input[i])
        #plt.imshow(full_inputs_test[i+n])
    '''  
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'cnn_train')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
        
    
    for i in range(n):
        test_directory = os.path.join(final_directory, r'cnn_train'+str(i) +'/')
        #test_directory = os.path.join(final_directory, r'test'+str(i+n) +'/')
        if not os.path.exists(test_directory):
            os.makedirs(test_directory)
        
        imwrite(str(test_directory)+ 'pred_mask.png', (np.clip(combined_masks[i],0,1)*255).astype(np.uint8))
        imwrite(str(test_directory) +'input.png', (combined_input[i]*255).astype(np.uint8))
        imwrite(str(test_directory) +'pred_mask_TH.png', (thresh_masks[i]*255).astype(np.uint8))
        imwrite(str(test_directory) +'GT_mask.png', (combined_GT_masks[i]*255).astype(np.uint8))
    
    n = 15
    n_patches = n * (512//roi)**2 
    
    test_mask = sess.run(pred_mask_test, feed_dict={u: full_test_set[0:n_patches], s: masks_test, is_training:False})
    print(test_mask.shape)
    
    combined_masks = np.zeros((n, 512,512))
    combined_GT_masks = np.zeros((n, 512,512))
    combined_input = np.zeros((n, 512,512,3))

    for i in range(n):
        combined_masks[i] = combine_img(test_mask[i*size*size:(i+1)*size*size,:,:,0], size, roi)
        combined_GT_masks[i] = combine_img(full_masks_test[size*size*(i):size*size*(i+1),:,:], size, roi)
        combined_input[i] = combine_img(full_test_set[size*size*(i):size*size*(i+1),:,:,3:6], size, roi)

    thresh_masks = np.ones_like(combined_masks)
    
    ind = np.where(combined_masks < 0.5)
    thresh_masks[ind] = 0.
    '''
    for i in range(n):
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(combined_GT_masks[i], cmap="gray")
        plt.subplot(2,2,2)
        plt.imshow(combined_masks[i], cmap="gray")
        plt.subplot(2,2,3)
        plt.imshow(thresh_masks[i], cmap="gray")
        plt.subplot(2,2,4)
        plt.imshow(combined_input[i])
        #plt.imshow(full_inputs_test[i+n])
    '''
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'cnn_tests')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
        
    
    for i in range(n):
        test_directory = os.path.join(final_directory, r'cnn_tests'+str(i) +'/')
        #test_directory = os.path.join(final_directory, r'test'+str(i+n) +'/')
        if not os.path.exists(test_directory):
            os.makedirs(test_directory)
        
        imwrite(str(test_directory)+ 'pred_mask.png', (np.clip(combined_masks[i],0,1)*255).astype(np.uint8))
        imwrite(str(test_directory) +'input.png', (combined_input[i]*255).astype(np.uint8))
        imwrite(str(test_directory) +'pred_mask_TH.png', (thresh_masks[i]*255).astype(np.uint8))
        imwrite(str(test_directory) +'GT_mask.png', (combined_GT_masks[i]*255).astype(np.uint8))
