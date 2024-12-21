import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchsummary import summary
from torchvision import models
import torchvision.transforms as transforms
from PyQt5.QtGui import QPixmap
import numpy as np



def initial(self):
    self.device=torch.device('cpu')
    self.model=torch.load("./vgg19_bn.pth", map_location=self.device)
    self.transform= transforms.Compose(
    [
    transforms.RandomHorizontalFlip(p=0.5),#隨機水平翻轉圖像，概率為0.5。
    transforms.RandomVerticalFlip(p=0.5),#隨機垂直翻轉圖像，概率為0.5。
    transforms.RandomRotation(degrees=30),#隨機旋轉 -30° 到 30° 之間
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
def show_augmentation(image_paths,labels,transform):   
    augmentation_imgs=[]
    for path in image_paths:
        image=Image.open(path).convert("RGB")
        augmentation_img=transform(image)
        augmentation_imgs.append(augmentation_img.permute(1, 2, 0).numpy())
        #增強後的影像是 PyTorch 張量格式，需用 .permute(1, 2, 0) 將通道順序從 (C, H, W) 改為 (H, W, C)，然後轉為 NumPy 格式以供 Matplotlib 顯示。
    
    fig,axes =plt.subplots(3,3,figsize=(12,12))
    axes=axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(augmentation_imgs):
            ax.imshow(augmentation_imgs[i])  # 顯示增強圖像
            ax.set_title(f'{labels[i]} - {augmentation_imgs[i].shape[:2]}', fontsize=12)  # 顯示標題，並顯示圖片的大小
            ax.axis('on')  # 顯示坐標軸
            #ax.set_xticks([])  # 隱藏 x 軸刻度
            #ax.set_yticks([])  # 隱藏 y 軸刻度
            
            # 設置統一的 x 和 y 軸範圍（這裡以 0 到圖片寬度、高度為範圍，您可以根據需要調整）
            ax.set_xlim(0, augmentation_imgs[i].shape[1])
            ax.set_ylim(augmentation_imgs[i].shape[0], 0)  # y 軸從上到下

    plt.tight_layout()
    plt.show()    
    
def show_structure(model): 
    summary(model.to('cpu'), input_size=(3, 32, 32))
    
def show_loss_acc(acc_path,loss_path):
    fig,axes =plt.subplots(2,1,figsize=(12,12))
    axes=axes.flatten()
    images=[]
    images.append(Image.open(acc_path).convert("RGB"))
    images.append(Image.open(loss_path).convert("RGB"))
    for i,ax in enumerate(axes):
        ax.imshow(images[i])
        ax.axis("off")
        
    plt.tight_layout()
    plt.show()    
    
def show_inference(self):
    #model = models.vgg19_bn(num_classes=10)
    #model.load_state_dict(self.model)
    model=self.model
    model.eval() # 設置模型為評估模式
    img=Image.open(self.img_path).convert("RGB")
    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
    
    with torch.no_grad():
        outputs=model(img_tensor)
        prob=torch.nn.functional.softmax(outputs,dim=1)# 使用 Softmax 轉換為概率
    
    # 獲取每個類別的概率
    prob_s = prob.squeeze().cpu().numpy()  # 去除 batch 維度並轉換為 numpy
    max_prob_index = np.argmax(prob_s) # 獲取最大機率的index

    # 創建 QPixmap 物件
    Q_img = QPixmap(self.img_path)
    self.label_2.setPixmap(Q_img)
    self.label_2.setScaledContents(True) 
    self.label_3.setText("Predicted= "+self.labels[max_prob_index])
    
    plt.figure(figsize=(10,6))
    plt.bar(self.labels,prob_s)
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.title("Probability of each class")
    plt.show()