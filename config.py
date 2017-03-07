class Config:
    data_path = "./figures"
    model_path = "./figures/checkpoint/model_20.ckpt"
    output_path = "./results"

    img_size = 424
    adjust_size = 444
    train_size = 424
    img_channel = 1
    conv_channel_base = 64

    learning_rate = 0.0002
    beta1 = 0.5
    max_epoch = 200
    L1_lambda = 100
    save_per_epoch=5
