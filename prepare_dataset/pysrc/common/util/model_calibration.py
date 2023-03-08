import tensorflow as tf
import numpy as np
import copy

# ICU_MODEL/src/utils/andy/stats.py
def get_ECE(pred, label, M=10):
    label = label.reshape((-1,1))
    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=1)
        label = np.expand_dims(label, axis=1)
    if pred.shape[1] == 1:
        pred = np.hstack((1 - pred, pred))
        label = np.hstack((1- label, label))


    N = float(len(pred))

    confidence = np.max(pred, axis=1)
    accuracy = np.argmax(pred, axis=1) == np.argmax(label, axis=1)

    ece = 0
    for m in range(M):
        interval = (m/M, (m+1)/M)

        idx = (confidence > interval[0]) == (confidence <= interval[1])
        idx = np.squeeze(idx)

        if np.sum(idx) > 0:
            bin_confidence = confidence[idx]
            bin_accuracy = accuracy[idx]

            ece += ( np.sum(idx) / N ) * np.abs(np.mean(bin_accuracy) - np.mean(bin_confidence))

    return ece


def get_MCE(pred, label, M=10):
    label = label.reshape((-1,1))
    if len(pred.shape) == 1:
        pred = np.expand_dims(pred, axis=1)
        label = np.expand_dims(label, axis=1)
    if pred.shape[1] == 1:
        pred = np.hstack((1 - pred, pred))
        label = np.hstack((1- label, label))


    confidence = np.max(pred, axis=1)
    accuracy = np.argmax(pred, axis=1) == np.argmax(label, axis=1)

    mce = 0

    for m in range(M):
        interval = (m/M, (m+1)/M)

        idx = (confidence > interval[0]) == (confidence <= interval[1])
        idx = np.squeeze(idx)

        if np.sum(idx) > 0:
            bin_confidence = confidence[idx]
            bin_accuracy = accuracy[idx]

            candidate =  np.abs(np.mean(bin_accuracy) - np.mean(bin_confidence))
            if candidate >= mce:
                mce = candidate

    return mce

# https://github.com/markdtw/temperature-scaling-tensorflow/blob/master/temp_scaling.py
# ICU_MODEL/src/utils/andy/tf_utils.py

def temp_scaling(sess, logit, label, fig_name=None, mode='valid'):
    #  logit = tf.expand_dims(logit, axis=1)
    #  label = tf.cast(tf.expand_dims(label, axis=1), tf.int32)
    label = label.astype(np.int)
    target = tf.cast(tf.argmax(tf.cast(label, tf.float32), axis=1), tf.float32)
    n_label = len(np.unique(label))

    with tf.variable_scope('temp', reuse=tf.AUTO_REUSE):
        temperature = tf.get_variable(name='weight',
                                      shape=[1],
                                      dtype=tf.float32,
                                      initializer=tf.initializers.constant(1.5),
                                      trainable=True)

    if mode == 'valid':
        sess.run([temperature.initializer])

    logit_calib = tf.divide(logit, temperature)
    
    # if n_label == 2:
    #     pred_calib = tf.nn.sigmoid(logit_calib)
    #     pred = tf.nn.sigmoid(logit)
    # else:
    #     pred_calib = tf.nn.softmax(logit_calib)
    #     pred = tf.nn.softmax(logit)

    pred_calib = tf.nn.softmax(logit_calib)
    pred = tf.nn.softmax(logit)

    # loss
    if n_label == 2:
        nll_loss_op = tf.compat.v1.losses.sigmoid_cross_entropy(label, logit_calib)
    else:
        nll_loss_op = tf.losses.softmax(label, logit_calib)

    if mode == 'valid':
        org_nll_loss_op = tf.identity(nll_loss_op)
        org_nll_loss = sess.run(org_nll_loss_op)

        # optimizer
        optim = tf.contrib.opt.ScipyOptimizerInterface(nll_loss_op, options={'maxiter': 100})
        optim.minimize(sess)
    else:
        if n_label == 2:
            org_nll_loss = sess.run(tf.compat.v1.losses.sigmoid_cross_entropy(label, logit))
        else:
            org_nll_loss = sess.run(tf.losses.softmax(label, logit))

    temperature_value = sess.run(temperature)
    nll_loss = sess.run(nll_loss_op)

    pred = sess.run(pred)
    ece = get_ECE(pred, label)
    mce = get_MCE(pred, label)

    # if fig_name:
    #     textstr = '\n'.join((
    #         r'ECE = {:.3f}'.format(ece),
    #         r'MCE = {:.3f}'.format(mce)))
    #     stats.plot_calibration(None, pred, label, textstr)
    #     plt.savefig(fig_name + '_before.svg')
    # _, _, _, roc = stats.plot_roc_curve(None, label, pred, '_nolegend_')
    # _, _, _, pr = stats.plot_pr_curve(None, label, pred, '_nolegend_')

    # print('[{}] temperature scaling before'.format(mode))
    # print('ECE: {}, MCE: {}, NLL: {}, ROC: {}, PR: {}'.format(ece, mce, org_nll_loss, roc, pr))

    logit_calib = sess.run(logit_calib)
    pred_calib = sess.run(pred_calib)
    ece_calib = get_ECE(pred_calib, label)
    mce_calib = get_MCE(pred_calib, label)
    
    # if fig_name:
    #     textstr = '\n'.join((
    #         r'ECE = {:.3f}'.format(ece_calib),
    #         r'MCE = {:.3f}'.format(mce_calib)))
    #     stats.plot_calibration(None, pred_calib, label, textstr)
    #     plt.savefig(fig_name + '_after.svg')
    # _, _, _, roc_calib = stats.plot_roc_curve(None, label, pred_calib, '_nolegend_')
    # _, _, _, pr_calib = stats.plot_pr_curve(None, label, pred_calib, '_nolegend_')

    # print('[{}] temperature scaling after: {}'.format(mode, temperature_value))
    # print('ECE: {}, MCE: {}, NLL: {}, ROC: {}, PR: {}'.format(ece_calib, mce_calib, nll_loss, roc_calib, pr_calib))

    is_correct_calib = tf.equal(tf.cast(tf.argmax(pred_calib, axis=1), tf.float32), target)
    accuracy_calib = tf.reduce_mean(tf.cast(is_correct_calib, tf.float32))
    cost_calib = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit_calib))

    cost_calib = sess.run(cost_calib)
    accuracy_calib = sess.run(accuracy_calib)

    return temperature_value, [cost_calib, accuracy_calib, logit_calib, pred_calib]


def platt_scaling(sess, logit, label, fig_name=None, mode='valid'):
    #  logit = tf.expand_dims(logit, axis=1)
    #  label = tf.cast(tf.expand_dims(label, axis=1), tf.int32)
    label = label.reshape((-1,))
    label = label.astype(np.float32)
    target = tf.cast(label, tf.float32)

    with tf.variable_scope('platt', reuse=tf.AUTO_REUSE):
        weight = tf.get_variable(name='weight',
                                 shape=[1],
                                 dtype=tf.float32,
                                 initializer=tf.initializers.constant(1.5),
                                 trainable=True)
        bias = tf.get_variable(name='bias',
                               shape=[1],
                               dtype=tf.float32,
                               initializer=tf.initializers.constant(1.5),
                               trainable=True)

    mapped_pos_label = (np.sum(label == 1) + 1) / (np.sum(label == 1) + 2)
    mapped_neg_label = 1 / (np.sum(label == 0) + 2)
    mapped_label = copy.deepcopy(label).astype(np.float32)
    mapped_label[label == 1] = mapped_pos_label
    mapped_label[label == 0] = mapped_neg_label

    if mode == 'valid':
        sess.run([weight.initializer, bias.initializer])

    logit_calib = tf.squeeze(
        tf.nn.bias_add(tf.expand_dims(tf.divide(logit, weight), axis=1), bias, data_format='NC...'))
    pred_calib = tf.nn.sigmoid(logit_calib)
    pred = tf.nn.sigmoid(logit)

    # loss
    nll_loss_op = tf.compat.v1.losses.sigmoid_cross_entropy(label, logit_calib)
    nll_loss_mapped_op = tf.compat.v1.losses.sigmoid_cross_entropy(mapped_label, logit_calib)

    if mode == 'valid':
        org_nll_loss_op = tf.identity(nll_loss_op)
        org_nll_loss_mapped_op = tf.identity(nll_loss_mapped_op)
        org_nll_loss = sess.run(org_nll_loss_op)
        org_nll_loss_mapped = sess.run(org_nll_loss_mapped_op)

        # optimizer
        optim = tf.contrib.opt.ScipyOptimizerInterface(nll_loss_mapped_op, options={'maxiter': 100})
        optim.minimize(sess)
    else:
        org_nll_loss = sess.run(tf.compat.v1.losses.sigmoid_cross_entropy(label, logit))
        org_nll_loss_mapped = sess.run(tf.compat.v1.losses.sigmoid_cross_entropy(mapped_label, logit))

    weight, bias = sess.run([weight, bias])
    nll_loss = sess.run(nll_loss_op)
    nll_loss_mapped = sess.run(nll_loss_mapped_op)

    pred = sess.run(pred)
    ece = get_ECE(pred, label)
    mce = get_MCE(pred, label)
    
    # if fig_name:
    #     textstr = '\n'.join((
    #         r'ECE = {:.3f}'.format(ece),
    #         r'MCE = {:.3f}'.format(mce)))
    #     stats.plot_calibration(None, pred, label, textstr)
    #     plt.savefig(fig_name + '_before.svg')
    # _, _, _, roc = stats.plot_roc_curve(None, label, pred, '_nolegend_')
    # _, _, _, pr = stats.plot_pr_curve(None, label, pred, '_nolegend_')

    # print('[{}] platt scaling before'.format(mode))
    # print('ECE: {}, MCE: {}, NLL_map: {}, NLL: {}, ROC: {}, PR: {}'.format(ece, mce, org_nll_loss_mapped, org_nll_loss, roc, pr))

    logit_calib = sess.run(logit_calib)
    pred_calib = sess.run(pred_calib)
    ece_calib = get_ECE(pred_calib, label)
    mce_calib = get_MCE(pred_calib, label)
    
    # if fig_name:
    #     textstr = '\n'.join((
    #         r'ECE = {:.3f}'.format(ece_calib),
    #         r'MCE = {:.3f}'.format(mce_calib)))
    #     stats.plot_calibration(None, pred_calib, label, textstr)
    #     plt.savefig(fig_name + '_after.svg')
    # _, _, _, roc_calib = stats.plot_roc_curve(None, label, pred_calib, '_nolegend_')
    # _, _, _, pr_calib = stats.plot_pr_curve(None, label, pred_calib, '_nolegend_')

    # print('[{}] platt scaling after: {}, {}'.format(mode, weight, bias))
    # print('ECE: {}, MCE: {}, NLL_map: {}, NLL: {}, ROC: {}, PR: {}'.format(ece_calib, mce_calib, nll_loss_mapped, nll_loss, roc_calib, pr_calib))
    is_correct_calib = tf.equal(tf.cast(pred_calib > 0.5, tf.float32), target)
    accuracy_calib = tf.reduce_mean(tf.cast(is_correct_calib, tf.float32))
    cost_calib = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit_calib))

    cost_calib = sess.run(cost_calib)
    accuracy_calib = sess.run(accuracy_calib)

    return weight, bias, [cost_calib, accuracy_calib, logit_calib, pred_calib]
