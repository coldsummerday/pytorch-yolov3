from  yolonet.data.dataset import VOCDetectionSet
from torchvision.transforms import  ToPILImage
from PIL import  Image,ImageDraw
import torch.utils.data as Data
labels = ["fenda", "yingyangkuaixian", "jiaduobao", "maidong","TYCL", "BSS", "TYYC", "LLDS", "KSFH", "MZY"]
data_root = "/home/zhou/data/voc_0221/"
data = VOCDetectionSet(data_root,labels)
loader = Data.DataLoader(
    dataset=data,      # torch TensorDataset format
    batch_size=8,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
)
img,label = loader.__iter__().__next__()
img_tensor,label_array = data[6]
tp = ToPILImage()
ori_img = tp(img_tensor)

ori_width, ori_height = ori_img.size
draw_handle = ImageDraw.Draw(ori_img)
for box_line in label_array:
    if box_line[0]==0:
        break
    x, y, w, h, label_index = box_line
    xmin = int((x - w / 2) * ori_width)
    ymin = int((y - h / 2) * ori_height)
    xmax = int((x + w / 2) * ori_width)
    ymax = int((y + h / 2) * ori_height)
    label = labels[int(label_index)]
    draw_handle.rectangle([(xmin, ymin), (xmax, ymax)])
    draw_handle.text((xmin, ymin - 10), label)

ori_img.show()



