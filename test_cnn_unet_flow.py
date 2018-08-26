import os
import numpy as np
from imageio import imread, imwrite
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import tensorflow as tf
from collections import OrderedDict

from image_utils import combine_img, threshold_mean, mirror_combine_data, reverse_flow_combine_data, generate_set_from_idx
from tf_utils import weight_variable, bias_variable, conv2d, deconv2d, weight_variable_devonc, crop_and_concat, max_pool, crop
from io_utils import read_masks, read_inputs

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
        
    #roi = 64
    roi = 128
    size = np.int(512 / roi)
    
    padding_width = 20
    
    masks_all = np.zeros((len(masks)*size*size, roi+2*padding_width, roi+2*padding_width))
    inputs_all = np.zeros((len(inputs)*size*size, roi+2*padding_width, roi+2*padding_width, 3))
    previous_all = np.zeros((len(inputs)*size*size, roi+2*padding_width, roi+2*padding_width, 3))
    nexts_all = np.zeros((len(inputs)*size*size, roi+2*padding_width, roi+2*padding_width, 3))

    masks_all = read_masks(masks, size, roi, padding_width)
    inputs_all = read_inputs(inputs, size, roi, padding_width)
    previous_all = read_inputs(previouss, size, roi, padding_width)
    nexts_all = read_inputs(nexts, size, roi, padding_width)
    
    
    

    full_set_all = np.zeros((inputs_all.shape[0], inputs_all.shape[1], inputs_all.shape[2], 9))
    full_set_all[:,:,:,0:3] = previous_all
    full_set_all[:,:,:,3:6] = inputs_all
    full_set_all[:,:,:,6:9] = nexts_all
    
    num_items = full_set_all.shape[0]
    ind_train = np.int64(num_items * (1-(1/4.8)))
    
    full_masks_test = masks_all[ind_train:]
    full_test_set = full_set_all[ind_train:]
    
    ind = threshold_mean(full_masks_test, 0.01, 0.8)
    
    masks_all = (~masks_all.astype(bool)).astype(int)
    full_masks_test = (~full_masks_test.astype(bool)).astype(int)
   
    
    test_set = full_test_set[ind]
    test_mask = full_masks_test[ind]
    
    print(full_set_all.shape, full_test_set.shape)
    print(masks_all.shape, full_masks_test.shape)
    
    
    test_set_large_area = generate_set_from_idx(full_test_set, np.array([1,2,3,12,13,14,16,19]), size)
    test_set_small_area = generate_set_from_idx(full_test_set, np.array([4,6,7,8,9,10,11,15,17,18]), size)
    test_set_large_flow = generate_set_from_idx(full_test_set,np.array([2,3,5,11,13,16,17,18]), size)
    test_set_small_flow = generate_set_from_idx(full_test_set,np.array([1,4,6,7,8,10,15,19]), size)
    
    masks_test_large_area = generate_set_from_idx(full_masks_test, np.array([1,2,3,12,13,14,16,19]), size)
    masks_test_small_area = generate_set_from_idx(full_masks_test, np.array([4,6,7,8,9,10,11,15,17,18]), size)
    masks_test_large_flow = generate_set_from_idx(full_masks_test,np.array([2,3,5,11,13,16,17,18]), size)
    masks_test_small_flow = generate_set_from_idx(full_masks_test,np.array([1,4,6,7,8,10,15,19]), size)

    #tensorflow
    
    
    
    n_channels = 9
    layers = 3
    features_root = 64
    filter_size = 3
    keep_prob = 1.
    pool_size = 2
    
    s = tf.placeholder(tf.float32, shape = [None, masks_all.shape[1], masks_all.shape[2]])
    u = tf.placeholder(tf.float32, shape = [None, full_set_all.shape[1], full_set_all.shape[1], n_channels])
        
    s_img = tf.reshape(s, [-1, masks_all.shape[1], masks_all.shape[2], 1])
    u_img = tf.reshape(u, [-1, full_set_all.shape[1], full_set_all.shape[2], n_channels])
        
    in_node = u_img
 
    weights = []
    biases = []
    convs = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()
    
    
    # down layers
    for layer in range(0, layers):
        features = 2**layer*features_root
        stddev = np.sqrt(2 / (filter_size**2 * features))
        if layer == 0:
            w1 = weight_variable([filter_size, filter_size, n_channels, features], stddev)
        else:
            w1 = weight_variable([filter_size, filter_size, features//2, features], stddev)
            
        w2 = weight_variable([filter_size, filter_size, features, features], stddev)
        b1 = bias_variable([features])
        b2 = bias_variable([features])
        
        conv1 = conv2d(in_node, w1, keep_prob)
        tmp_h_conv = tf.nn.relu(conv1 + b1)
        conv2 = conv2d(tmp_h_conv, w2, keep_prob)
        dw_h_convs[layer] = tf.nn.relu(conv2 + b2)
        
        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))

        if layer < layers-1:
            pools[layer] = max_pool(dw_h_convs[layer], pool_size)
            in_node = pools[layer]
        
    in_node = dw_h_convs[layers-1]
        
    # up layers
    for layer in range(layers-2, -1, -1):
        features = 2**(layer+1)*features_root
        stddev = np.sqrt(2 / (filter_size**2 * features))
        
        wd = weight_variable_devonc([pool_size, pool_size, features//2, features], stddev)
        bd = bias_variable([features//2])
        h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
        h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
        deconv[layer] = h_deconv_concat
        
        w1 = weight_variable([filter_size, filter_size, features, features//2], stddev)
        w2 = weight_variable([filter_size, filter_size, features//2, features//2], stddev)
        b1 = bias_variable([features//2])
        b2 = bias_variable([features//2])
        
        conv1 = conv2d(h_deconv_concat, w1, keep_prob)
        h_conv = tf.nn.relu(conv1 + b1)
        conv2 = conv2d(h_conv, w2, keep_prob)
        in_node = tf.nn.relu(conv2 + b2)
        up_h_convs[layer] = in_node

        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))


    # Output Map
    weight = weight_variable([1, 1, features_root, 1], stddev)
    bias = bias_variable([1], -0.5)
    conv = conv2d(in_node, weight, tf.constant(1.0))
    pred_mask = tf.nn.sigmoid(conv + bias)
    
    # loss

    #loss = tf.losses.mean_squared_error(pred_mask, crop(s_img,pred_mask))
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=s_, labels=s))  
    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=s_, labels=s))     
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=conv+bias, labels=crop(s_img,pred_mask))) #sigmoid 
    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_mask, labels=crop(s_img,pred_mask))) #relu
    loss_sum = tf.summary.scalar("cross_entropy_unet", loss)
    
    
    loss_batch = tf.placeholder(tf.float32)
    loss_batch_summary = tf.summary.scalar("cross_entropy_unet", loss_batch)
    
    
    #accuracy
    
    #correct_prediction = tf.equal(tf.round(tf.clip_by_value(pred_mask, 0., 1.),),crop(s_img,pred_mask)) # relu
    correct_prediction = tf.equal(tf.round(pred_mask),crop(s_img,pred_mask))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
    accuracy_sum = tf.summary.scalar("accuracy_unet", accuracy)
    
    accuracy_batch = tf.placeholder(tf.float32)
    accuracy_batch_summary = tf.summary.scalar("accuracy_unet", accuracy_batch)
    
    
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
    #sess.run(init)
    #train_writer = tf.summary.FileWriter( './logs/1/train ', sess.graph)
    
   

    '''
    batch_size = 15
    n_batches = int(np.ceil(training_set.shape[0] / batch_size))
    
    inputs_train_batch = np.array_split(training_set, n_batches, axis = 0)
    masks_train_batch = np.array_split(masks_train, n_batches, axis = 0)
    '''
    saver = tf.train.Saver()
    current_directory = os.getcwd()
    model_directory = os.path.join(current_directory, r'unet')
    #if not os.path.exists(model_directory):
    #    os.makedirs(model_directory)
        
    saver.restore(sess, model_directory+"/model")
    f = open("unet.txt", "w")
    test_loss, test_acc, test_summary = sess.run([loss, accuracy, merged], feed_dict={u: test_set, s: test_mask})
    print("Test Set: Loss:", test_loss, "Acc:", test_acc)
    f.write("Test Set: Loss: " + str(test_loss) + " Acc: " + str(test_acc) + "\n")
      
    test_loss, test_acc, test_LA_summary = sess.run([loss, accuracy, merged], feed_dict={u: test_set_large_area, s: masks_test_large_area})
    print("LA: Loss:", test_loss, "Acc:", test_acc)
    f.write("LA: Loss: " + str(test_loss) + " Acc: " + str(test_acc) + "\n")
    
    test_loss, test_acc, test_SA_summary = sess.run([loss, accuracy, merged], feed_dict={u: test_set_small_area, s: masks_test_small_area})
    print("SA: Loss:", test_loss, "Acc:", test_acc)
    f.write("SA: Loss: " + str(test_loss) + " Acc: " + str(test_acc) + "\n")
    
    test_loss, test_acc, test_LF_summary = sess.run([loss, accuracy, merged], feed_dict={u: test_set_large_flow, s: masks_test_large_flow})
    print("LF: Loss:", test_loss, "Acc:", test_acc)
    f.write("LF: Loss: " + str(test_loss) + " Acc: " + str(test_acc) + "\n")
    
    test_loss, test_acc, test_SF_summary = sess.run([loss, accuracy, merged], feed_dict={u: test_set_small_flow, s: masks_test_small_flow})
    print("SF: Loss:", test_loss, "Acc:", test_acc)
    f.write("SF: Loss: " + str(test_loss) + " Acc: " + str(test_acc) + "\n")
    
      
    '''   
    n = 15
    n_patches = n * (512//roi)**2 
    
    test_mask = sess.run(pred_mask, feed_dict={u: full_test_set[0:n_patches], s: full_masks_test})
    print(test_mask.shape)
    
    combined_masks = np.zeros((n, 512,512))
    combined_GT_masks = np.zeros((n, 512,512))
    combined_input = np.zeros((n, 512,512,3))

    for i in range(n):
        combined_masks[i] = combine_img(test_mask[i*size*size:(i+1)*size*size,:,:,0], size, roi)
        combined_GT_masks[i] = combine_img(full_masks_test[size*size*(i):size*size*(i+1),padding_width:-1-padding_width+1,padding_width:-1-padding_width+1], size, roi)
        combined_input[i] = combine_img(full_test_set[size*size*(i):size*size*(i+1),padding_width:-1-padding_width+1,padding_width:-1-padding_width+1,3:6], size, roi)

    thresh_masks = np.ones_like(combined_masks)
    
    ind = np.where(combined_masks < 0.5)
    thresh_masks[ind] = 0.

    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'cnn_tests_restore')
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
    '''