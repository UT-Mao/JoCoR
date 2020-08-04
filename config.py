class Config:
    #dataset
    noise_rate = 0.2

    # param for schedule
    n_epoch = 200
    epoch_decay_start = 80

    #param for optimizer
    learning_rate = 0.001
    mom1 = 0.9
    mom2 = 0.1
    #for rate schedule
    num_gradual = 10

    noise_rate = 0.2
    forget_rate = noise_rate
    exponent = 1

    load = False
    cnn1_path = None
    cnn2_path = None

    save = True
    save_path_1 = './cnn1.pth'
    save_path_2 = './cnn2.pth'



opt = Config()
