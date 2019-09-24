import tensorflow as tf
from projects.paper9_charcnn.get_data import data_load
from projects.paper9_charcnn.model import model
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
data_load = data_load()
model = model(num_classes=data_load.num_classes)
session_config = tf.ConfigProto(
                    log_device_placement=False,
                    inter_op_parallelism_threads=0,
                    intra_op_parallelism_threads=0,
                    allow_soft_placement=True)
# session_config.gpu_options.allow_growth = True
# session_config.gpu_options.allocator_type = 'BFC'
with tf.Session(config = session_config) as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    best_dev_acc = 0
    biggest_patient = 5
    patient = 0
    for i in range(10):
        model.train(sess,data_load.train_datas,data_load.train_labels,32)
        dev_acc = model.test(sess,data_load.dev_datas,data_load.dev_labels,32)
        if dev_acc>best_dev_acc:
            best_dev_acc = dev_acc
            saver.save(sess,"./model/best_result.ckpt")
            patient=0
            print ("Epoch %d: best dev acc is updataed to %f"%(i,best_dev_acc))
        else:
            patient+=1
            print ("Epoch %d: best acc is not updataed, the patient is %d"%(i,patient))
        if patient==biggest_patient:
            print ("Patient is achieve biggest patient, training finished")
    saver.restore(sess,"./model/best_result.ckpt")
    test_acc = model.test(sess,data_load.test_datas,data_load.test_labels,32)
    print ("Test acc in best dev acc is:",test_acc)
