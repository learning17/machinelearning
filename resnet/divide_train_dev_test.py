import random

if __name__ == '__main__':
    file_dir = "/data/ImageNet/ILSVRC2012_img_train.txt"
    train_file = "/data/ImageNet/ILSVRC2012_img_train/train.txt"
    dev_file = "/data/ImageNet/ILSVRC2012_img_train/dev.txt"
    val_ratio = 0.05
    step = 0
    with open(file_dir,'r') as f, open(train_file,'w') as trainf, open(dev_file,'w') as devf:
        for line in f.readlines():
            line = line.strip()
            ran = random.random()
            print(step)
            step += 1
            if ran <= val_ratio:
                devf.write(line + '\n')
            else:
                trainf.write(line + '\n')

