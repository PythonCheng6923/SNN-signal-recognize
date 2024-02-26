import os
import struct
import numpy as np
import re
from sklearn.preprocessing import StandardScaler


save_path = r'D:\pythonProject\Python Workspace\SNN信号调制识别\任务代码和数据集\dataset\任务'
path = r'D:\pythonProject\Python Workspace\SNN信号调制识别\任务代码和数据集\dataset\任务\test'
file_list = os.listdir(path)

classes = ["CDMA ", "DTMB", "FM广播", "GSM", "LTE", "NB-IOT", "指点信标", "TETRA", "WCDMA", "5G NR",
           "未知信号002","未知信号001"]
print(len(classes))

# 分割
x_train = []
x_test = []
y_train = []
y_test = []
snrs_train = []
snrs_test = []

print("该文件夹下的文件数", len(file_list))
file_number = 1
# 只取0~1999
# for file_name in file_list[:2000]:
# 取2000~
# for file_name in file_list[2000:]:
# 全跑
for file_name in file_list:
    file_path = os.path.join(path, file_name)
    if os.path.isfile(file_path):
        with open(file_path, "rb") as f:

            IQ_SNR = int(float(file_path.split("_")[-1].replace('.sample', '')))
            # print("IQ_SNR", IQ_SNR)
            data = f.read()
            # 读取数据信息，包括MagicNumber、数据格式、数据组织方式、采样率、信号中心频率、分类类型、分类标签、数据点数量、IQ数据
            # print("MagicNumber", data[0:6])
            # print("数据格式", data[6])
            # print("数据组织方式", data[7])
            data3 = struct.unpack("d", data[8:8+8])
            # print("采样率", data3)

            data4 = struct.unpack("d", data[8+8:8+8+8])
            # print("信号中心频率", data4)

            # print("分类类型", data[8+8+8])
            # utf-8报错修正.decode('utf-8','ignore')
            label = data[8 + 8 + 8 + 1:8 + 8 + 8 + 1 + 100].replace(b'\x00', b'').decode('utf-8','ignore') #去除NULL，获得制式字符
            # print("分类标签:", label)
            # 只取体制lable缩写
            match = re.search(r'(.+?)[（(]', label)
            # Nonetype没有group，故加判断
            if match != None:
                label = match.group(1)
            print("分类标签:", label, end = '')
            label_onehot = [0] * len(classes)
            label_onehot[classes.index(label)] = 1
            # print("One-Hot分类标签:", label_onehot)

            data5 = struct.unpack("i"*2, data[8+8+8+1+100:8+8+8+1+100+8])
            # print("数据点数量", data5)

            # print("IQ数据占字节数", len(data[8+8+8+1+100+8:]))

            data6 = struct.unpack("f" * data5[0] * 2, data[8+8+8+1+100+8:])
            # print("I和Q数据总数", len(data6))
            # print("原类型为", type(data6))
            data6 = list(data6)
            # print("转换类型为", type(data6))
            # print("I和Q数据总数", len(data6))
            
            # 针对每一个文件进行标准化处理
            # 创建一个标准化处理器对象
            scaler = StandardScaler()
            # 将数据转换为二维数组形式，并进行标准化处理
            normalized_data = scaler.fit_transform(np.array(data6).reshape(-1, 1))
            # 输出标准化后的数据
            data6 = normalized_data.flatten()
            print("已标准化")
            # print("标准化之后：", data6[:10])
            

            I_list = [data6[2 * i] for i in range(data5[0])]
            Q_list = [data6[2* i + 1] for i in range(data5[0])]
            # print("I数据总数", len(I_list))
            # print("Q数据总数", len(Q_list))

            # print("I数据前1024", I_list[:1024])
            # print("Q数据前1024", Q_list[:1024])

            # IQ数据划分，以81920为一信号样本单位，并把IQ合并（81920，2），然后加入样本数目（样本数，81920，2），最后转为（样本数，1，81920，2）
            num_samples = len(I_list) // 10240
            print("该文件下的大小为10240个数据点的信号样本数目", num_samples)
            IQ_samples = []
            IQ_labels = []
            IQ_snrs = []

            for i in range(num_samples):
                start_idx = i * 10240
                end_idx = (i+1) * 10240
                I_sample = I_list[start_idx:end_idx]
                Q_sample = Q_list[start_idx:end_idx]

                IQ_sample = np.column_stack((I_sample, Q_sample))
                IQ_samples.append(IQ_sample)
                IQ_labels.append(label_onehot)  # one-hot label
                IQ_snrs.append(IQ_SNR)

            IQ_samples_array = np.stack(IQ_samples)
            IQ_labels_array = np.stack(IQ_labels)
            IQ_snrs_array = np.array(IQ_snrs)

            # print("合并后的IQ信号样本尺寸", IQ_samples_array.shape)
            # print("合并后的IQ信号样本：", IQ_samples_array)
            # print("对应的label尺寸", IQ_labels_array.shape)
            # print("对应的label", IQ_labels_array)
            # print("对应的SNR尺寸", IQ_snrs_array.shape)

            # resize
            IQ_samples_data = np.expand_dims(IQ_samples_array, axis=1)
            # print("转换后的IQ数据尺寸：", IQ_samples_data.shape)
            # print("转换后的IQ数据：", IQ_samples_data)
            x_test.append(IQ_samples_data)
            y_test.append(IQ_labels_array)
            snrs_test.append(IQ_snrs_array)
            print("第" + str(file_number) + "个文件采集完毕")
            file_number = file_number + 1


x_test = np.concatenate(x_test)
y_test = np.concatenate(y_test)
snrs_test = np.concatenate(snrs_test)


print("总x_test的尺寸：", x_test.shape)
print("总y_test的尺寸：", y_test.shape)
print("总snrs_test的尺寸：", snrs_test.shape)

# save
if not(os.path.exists(save_path + "/data")):
    os.mkdir(save_path + "/data")

np.save(save_path + "/data/x_test.npy", x_test)
np.save(save_path + "/data/y_test.npy", y_test)
np.save(save_path + "/data/snrs_test.npy", snrs_test)
print(save_path + "成功保存")
