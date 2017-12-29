# coding=utf8


import tensorflow as tf
import super_resolution as sr
import super_resolution_utilty as util

flags = tf.app.flags
FLAGS = flags.FLAGS

# Model
flags.DEFINE_float("initial_lr", 0.001, "初始化学习率")
flags.DEFINE_float("lr_decay", 0.5, "Learning rate decay rate when it does not reduced during specific epoch")
flags.DEFINE_integer("lr_decay_epoch", 4, "4个epoch做一次学习率衰减 当损失不减少时，衰减学习率")
flags.DEFINE_float("beta1", 0.1, "Beta1 形式 adam optimizer")
flags.DEFINE_float("beta2", 0.1, "Beta2 形式 adam optimizer")
flags.DEFINE_float("momentum", 0.9, "Momentum for momentum optimizer and rmsprop optimizer")

flags.DEFINE_integer("feature_num", 96, "卷积核数量")
flags.DEFINE_integer("cnn_size", 3, "卷积核大小")
flags.DEFINE_integer("inference_depth", 9, "递归层的个数")
flags.DEFINE_integer("batch_num", 64, "用于训练的mini-batch的图片数量")
flags.DEFINE_integer("batch_size", 41, "mini-batch图像大小")
flags.DEFINE_integer("stride_size", 21, "mini-batch的步长 隔21个数据取mini-batch")
flags.DEFINE_string("optimizer", "adam", "优化器:  [gd, momentum, adadelta, adagrad, adam, rmsprop]")
flags.DEFINE_float("loss_alpha", 1, "初始化 loss-alpha 值 (0-1). 为0时不使用中间输出") #最初a
flags.DEFINE_integer("loss_alpha_zero_epoch", 25, "Decrease loss-alpha to zero by this epoch") #随训练进行a衰退
flags.DEFINE_float("loss_beta", 0.0001, "Loss-beta for 权重衰减")
flags.DEFINE_float("weight_dev", 0.001, "初始化 weight 标准差")
flags.DEFINE_string("initializer", "he", "初始化方法: can be [uniform, stddev, diagonal, xavier, he]")

# Image Processing
flags.DEFINE_integer("scale", 2, "Scale for Super Resolution (can be 2 or 4)")
flags.DEFINE_float("max_value", 255.0, "用于规格化图像像素值")
flags.DEFINE_integer("channels", 1, "Using num of image channels. Use YCbCr when channels=1.")
flags.DEFINE_boolean("jpeg_mode", False, "使用Jpeg模式从rgb转换为ycbcr")
flags.DEFINE_boolean("residual", False, "不使用残差网络")

# Training or Others
flags.DEFINE_boolean("is_training", True, "用91个标准图像训练模型")
flags.DEFINE_string("dataset", "set5", "测试数据集. [set5, set14, bsd100, urban100, all, test] are available")
flags.DEFINE_string("training_set", "ScSR", "训练数据集. [ScSR, Set5, Set14, Bsd100, Urban100] are available")
flags.DEFINE_integer("evaluate_step", 20, "经多少次评估一下")
flags.DEFINE_integer("save_step", 2000, "经多少次保存一下训练模型")
flags.DEFINE_float("end_lr", 1e-5, "训练结束的学习率")
flags.DEFINE_string("checkpoint_dir", "model", "Directory for checkpoints")
flags.DEFINE_string("cache_dir", "cache", "用于缓存图像数据的目录。 如果指定，建立图像缓存")
flags.DEFINE_string("data_dir", "data", " test/train images目录")
flags.DEFINE_boolean("load_model", False, "开始前加载保存的模型")
flags.DEFINE_string("model_name", "", "model name for save files and tensorboard log")

# Debugging or Logging调试或记录
flags.DEFINE_string("output_dir", "output", "输出测试图像目录")
flags.DEFINE_string("log_dir", "tf_log", " tensorboard log目录")
flags.DEFINE_boolean("debug", False, "显示每个计算的MSE和权重变量")
flags.DEFINE_boolean("initialise_log", True, "Clear all tensorboard log before start")
flags.DEFINE_boolean("visualize", True, "保存loss和graph data")
flags.DEFINE_boolean("summary", False, "保存w和b")


def main(_):

  #print("Super Resolution (tensorflow version:%s)" % tf.__version__)
  print("%s\n" % util.get_now_date()) #开始时间

  if FLAGS.model_name is "": #保存模型名字
    model_name = "model_F%d_D%d_LR%f" % (FLAGS.feature_num, FLAGS.inference_depth, FLAGS.initial_lr)
  else:
    model_name = "model_%s" % FLAGS.model_name 
    
  model = sr.SuperResolution(FLAGS, model_name=model_name) #建立DRCN模型 

  test_filenames = util.build_test_filenames(FLAGS.data_dir, FLAGS.dataset, FLAGS.scale) #获取测试文件
  if FLAGS.is_training:
    if FLAGS.dataset == "test":
      training_filenames = util.build_test_filenames(FLAGS.data_dir, FLAGS.dataset, FLAGS.scale)
    else:
      training_filenames =  util.get_files_in_directory(FLAGS.data_dir + "/" + FLAGS.training_set + "/") #获得训练文件

    print("Loading and building cache images...")
    model.load_datasets(FLAGS.cache_dir, training_filenames, test_filenames, FLAGS.batch_size, FLAGS.stride_size) #模型用的训练数据、测试数据、缓存图像数据目录、隔多少个数据取一次mini-batch
  else:
    FLAGS.load_model = True

  model.build_embedding_graph() # embedding层
  model.build_inference_graph() # inference层
  model.build_reconstruction_graph() # reconstruction层
  model.build_optimizer() #创建损失函数及优化器
  model.init_all_variables(load_initial_data=FLAGS.load_model) #初始化变量

  if FLAGS.is_training:
    train(training_filenames, test_filenames, model)
  
  psnr = 0
  total_mse = 0
  for filename in test_filenames:
    mse = model.do_super_resolution_for_test(filename, FLAGS.output_dir) # 对测试图片进行超分辨
    total_mse += mse
    psnr += util.get_psnr(mse)  #
 
  print ("\n--- summary --- %s" % util.get_now_date()) #结束时间
  model.print_steps_completed() #输出完成训练所需的总epoch，steps，time等
  util.print_num_of_total_parameters() #输出一共有多少参数
  print("Final MSE:%f, PSNR:%f" % (total_mse / len(test_filenames), psnr / len(test_filenames)))

  
def train(training_filenames, test_filenames, model):

  mse = model.evaluate() #初始状态 训练数据集的MSE
  model.print_status(mse) #初始状态 显示训练的epoch lr a等

  while model.lr > FLAGS.end_lr: #训练没达到终止状态
  
    logging = model.step % FLAGS.evaluate_step == 0
    model.build_training_batch() #建立mini-batch batch_input_images batch_true_images
    model.train_batch(log_mse=logging) #训练

    if logging: 
      mse = model.evaluate()
      model.print_status(mse)

    if model.step > 0 and model.step % FLAGS.save_step == 0:
      model.save_model() #保存模型 Model saved

  model.end_train_step() #训练花费总时间
  model.save_all() #保存所有模型及文件 Graph saved

  # if FLAGS.debug: #输出-1、0层权重及偏置，递归层权重
    # model.print_weight_variables()
    

if __name__ == '__main__':
  tf.app.run()
