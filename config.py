class Config:
    #used for training	
    data_path = "./figures"
    save_path = "./model"

    #if you are not going to train from the very beginning, change this path to the existing model path
    model_path = ""#./model/model.ckpt"

    start_epoch = 0
    output_path = "./result"

    #used GPU
    use_gpu = 1

    #changed to FITs, mainly refer to the size
    img_size = 424
    train_size = 424
    img_channel = 1
    conv_channel_base = 64

    #Scaling
    pixel_max_value = 700
    scale_factor = 100

    learning_rate = 0.0002
    beta1 = 0.5
    max_epoch = 20
    L1_lambda = 100
    sum_lambda = 0####
    save_per_epoch=1
