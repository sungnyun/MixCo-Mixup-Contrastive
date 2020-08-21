python main_mixco.py -a resnet18 --lr 0.015 --batch-size 128 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --exp-name contmix_to05 ./data/tiny-imagenet


python main_lincls.py -a resnet18 --lr 3.0 --batch-size 256 --pretrained contmix_to05.pth.tar --gpu 1 ./data/tiny-imagenet