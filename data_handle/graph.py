# encoding=utf-8
from matplotlib import pyplot
import matplotlib.pyplot as plt
 
names = range(8,21)
names = [str(x) for x in list(names)]
 
x = range(len(names))

# y_train = [0.840,0.839,0.834,0.832,0.824,0.831,0.823,0.817,0.814,0.812,0.812,0.807,0.805]
# y_test  = [0.838,0.840,0.840,0.834,0.828,0.814,0.812,0.822,0.818,0.815,0.807,0.801,0.796]
#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
#pl.xlim(-1, 11)  # 限定横轴的范围
#pl.ylim(-1, 110)  # 限定纵轴的范围
 
plt.figure(figsize = (9, 4))
plt.subplot(1, 2, 1)
 
vgg16    = [66.36, 66.50, 66.28, 65.57, 65.75, 64.23, 55.29]
resnet18 = [75.00, 75.63, 74.85, 75.30, 74.16, 73.56, 64.67]
resnet34 = [75.43, 75.23, 75.12, 74.85, 74.78, 74.14, 65.00]
alexnet  = [59.67, 57.59, 58.26, 57.17, 57.15, 55.57, 47.39]
x        = ['fp' ,'8bit','7bit','6bit','5bit','4bit','3bit']


vgg16_top5    = [87.50, 87.49, 87.53, 86.66, 86.38, 86.44, 80.70]
resnet18_top5 = [93.44, 93.33, 93.15, 93.08, 93.06, 92.84, 88.20]
resnet34_top5 = [93.03, 92.71, 93.10, 92.97, 93.36, 92.54, 88.05]
alexnet_top5  = [84.87, 84.87, 84.73, 84.50, 84.55, 83.59, 77.58]


# plt.figure(dpi = 200, figsize = (16, 9))
plt.plot(x, vgg16,"bo-", label = "vgg16")
plt.plot(x, resnet18, 'y*-', label = "resnet18")
plt.plot(x, resnet34, 'r+-', label = "resnet34")
plt.plot(x, alexnet, 'g^-', label = "alexnet")
plt.legend()
plt.xlabel("train method")
plt.ylabel("Top-1 accuracy")

plt.subplot(1, 2, 2)

plt.plot(x, vgg16_top5,"bo-", label = "vgg16")
plt.plot(x, resnet18_top5, 'y*-', label = "resnet18")
plt.plot(x, resnet34_top5, 'r+-', label = "resnet34")
plt.plot(x, alexnet_top5, 'g^-', label = "alexnet")
plt.legend()
plt.xlabel("train method")
plt.ylabel("Top-5 accuracy")


# plt.show()
plt.savefig("f1.png", dpi = 900)

# plt.plot(x, y_test, marker='*', ms=10,label='uniprot90_test')
# plt.legend()  # 让图例生效
# plt.xticks(x, names, rotation=1)
 
# plt.margins(0)
# plt.subplots_adjust(bottom=0.10)
# plt.xlabel('the length') #X轴标签
# plt.ylabel("f1") #Y轴标签
# pyplot.yticks([0.750,0.800,0.850])
# #plt.title("A simple plot") #标题
# plt.savefig('f1.png',dpi = 900)
