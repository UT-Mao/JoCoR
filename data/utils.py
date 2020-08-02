import numpy as np

def noisify_sym(label, noise_rate, random_state=None, num_classes=10):
    P = np.ones((num_classes, num_classes))
    n = noise_rate
    P = (n / (num_classes - 1)) * P
    for i in range(0, num_classes):
        P[i, i] = 1- n
    label_noisy = multiclass_noisify(label, P=P,random_state=random_state)
    actual_noise = (label_noisy != label).mean()
    print('Actual noise rate is', actual_noise)
    label = label_noisy

    return label, actual_noise

def multiclass_noisify(label, P, random_state=0):
    num = label.shape[0]
    new_label = label.copy()
    flipper = np.random.RandomState(random_state)
    for idx in np.arange(num):
        this_label = label[idx]
        flipped = flipper.multinomial(1, P[this_label], 1)[0]
        new_label[idx] = np.where(flipped == 1)[0]
    return new_label
