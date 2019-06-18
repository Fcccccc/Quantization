from matplotlib import pyplot
import matplotlib.pyplot as plt


def main():
    labels = [ "conv", "fc" ]
    Alexnet_size  = [ 2085412, 1677412 ]
    Renset18_size = [ 12543296, 51300 ]
    Resnet34_size = [ 22651456, 51300 ]
    Vgg16_size    = [ 10016960, 19308644 ]
    
    def subplot(pos, data, name):
        plt.subplot(pos)
        plt.pie(data, labels=labels,autopct='%1.1f%%',shadow=False,startangle=150)
        plt.title(name)
        plt.axis('equal')
    subplot(141, Alexnet_size, "Alexnet")
    subplot(142, Vgg16_size, "Vgg16")
    subplot(143, Renset18_size, "Resnet18")
    subplot(144, Resnet34_size, "Renset34")
    plt.show()
main()


