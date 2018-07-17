import tensorflow as tf


def cnn_model(input_images, batch_size, drop_out_rate=0.1, is_training=False, target_value=None):
    def truncated_normal_var(name, shape, dtype):
        return (tf.get_variable(name=name, shape=shape, dtype=dtype,
                                initializer=tf.truncated_normal_initializer(stddev=0.05)))

    def zero_var(name, shape, dtype):
        return (tf.get_variable(name=name, shape=shape, dtype=dtype, initializer=tf.constant_initializer(0.0)))

    with tf.variable_scope('conv1') as scope:
        conv1_kernel = truncated_normal_var(name='conv_kernel1', shape=[11, 11, 1, 64],
                                            dtype=tf.float32)
        conv1 = tf.nn.conv2d(input_images, conv1_kernel, [1, 3, 3, 1], padding='SAME')
        conv1_bias = truncated_normal_var(name='conv_bias1', shape=[64], dtype=tf.float32)
        conv1_add_bias = tf.nn.bias_add(conv1, conv1_bias)
        relu_conv1 = tf.nn.relu(conv1_add_bias)

    norm1 = tf.nn.lrn(relu_conv1, depth_radius=5, bias=2.5, alpha=1e-3, beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
        conv2_kernel = truncated_normal_var(name='conv_kernel2', shape=[5, 5, 64, 32], dtype=tf.float32)
        conv2 = tf.nn.conv2d(norm1, conv2_kernel, [1, 1, 1, 1], padding='SAME')
        conv2_bias = truncated_normal_var(name='conv_bias2', shape=[32], dtype=tf.float32)
        conv2_add_bias = tf.nn.bias_add(conv2, conv2_bias)
        relu_conv2 = tf.nn.relu(conv2_add_bias)

    norm2 = tf.nn.lrn(relu_conv2, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm2')

    with tf.variable_scope('conv3') as scope:
        conv3_kernel = truncated_normal_var(name='conv_kernel3', shape=[3, 3, 32, 32], dtype=tf.float32)
        conv3 = tf.nn.conv2d(norm2, conv3_kernel, [1, 1, 1, 1], padding='SAME')
        conv3_bias = truncated_normal_var(name='conv_bias3', shape=[32], dtype=tf.float32)
        conv3_add_bias = tf.nn.bias_add(conv3, conv3_bias)
        relu_conv3 = tf.nn.relu(conv3_add_bias)

    pool = tf.nn.max_pool(relu_conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_layer')
    norm3 = tf.nn.lrn(pool, depth_radius=5, bias=2.0, alpha=1e-3, beta=0.75, name='norm3')
    reshaped_output = tf.reshape(norm3, [batch_size, -1])
    reshaped_dim = reshaped_output.get_shape()[1].value

    with tf.variable_scope('full1') as scope:
        full_weight1 = truncated_normal_var(name='full_mult1', shape=[reshaped_dim, 384], dtype=tf.float32)
        full_bias1 = truncated_normal_var(name='full_bias1', shape=[384], dtype=tf.float32)
        full_layer1 = tf.nn.relu(tf.add(tf.matmul(reshaped_output, full_weight1), full_bias1))

    with tf.variable_scope('full2') as scope:
        full_weight2 = truncated_normal_var(name='full_mull2', shape=[384, 192], dtype=tf.float32)
        full_bias2 = truncated_normal_var(name='full_bias2', shape=[192], dtype=tf.float32)
        full_layer2 = tf.nn.relu(tf.add(tf.matmul(full_layer1, full_weight2), full_bias2))

    with tf.variable_scope('full3') as scope:
        full_weight3 = truncated_normal_var(name='full_mull3', shape=[192, len(target_value)], dtype=tf.float32)
        full_bias3 = truncated_normal_var(name='full_bias3', shape=[len(target_value)], dtype=tf.float32)
        final_output = tf.nn.relu(tf.add(tf.matmul(full_layer2, full_weight3), full_bias3))
    final_output = tf.layers.dropout(final_output, rate=drop_out_rate, training=is_training)

    return final_output