import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.uic import loadUi
from PIL import Image
import HW2_Q1
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Main_UI(QMainWindow):
    def __init__(self, parent = None):
        super(Main_UI,self).__init__(parent)
        loadUi("./HW2.ui",self)
        self.device=None
        self.model=None
        self.transform=None
        self.files=[]
        self.labels=['airplane', 'automobile', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.acc="./vgg19_bn_acc.png"
        self.loss="./vgg19_bn_loss.png"
        self.loadFiles=""
        self.img_path=""
        HW2_Q1.initial(self)
        self.Connect_btn()


    def Connect_btn(self):
        self.pushButton.clicked.connect(self.pushButton1F)           #load file
        self.pushButton_2.clicked.connect(self.pushButton2F)         #1. Show Augmented Images
        self.pushButton_3.clicked.connect(self.pushButton3F)         #2. Show Model Structure
        self.pushButton_4.clicked.connect(self.pushButton4F)         #3. Show Accuracy and Loss
        self.pushButton_5.clicked.connect(self.pushButton5F)         #4. Inference
        self.pushButton_6.clicked.connect(self.showMessage_wrong_btn)         #1. Show Training Images
        self.pushButton_7.clicked.connect(self.showMessage_wrong_btn)         #2. Show Model Structure
        self.pushButton_8.clicked.connect(self.showMessage_wrong_btn)         #3. Show Training Loss
        self.pushButton_9.clicked.connect(self.showMessage_wrong_btn)    #4. Inference

    
    def showMessage_wrong_btn(self):
        msg = QMessageBox()
        msg.setWindowTitle("操作錯誤提示")
        msg.setText("非此題按鈕")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
    
    def pushButton1F(self):
        img_path, _  = QFileDialog.getOpenFileName(self, "Select Image")
        #QFileDialog.getOpenFileName 返回的是一個元組 (file_path, file_filter)，其中 file_path 是選擇的文件的完整路徑，而 file_filter 是文件選擇框的過濾器（例如 *.png 或 *.jpg）。
        if not img_path:
            QMessageBox.warning(self, "操作錯誤提示", "請重新選擇影像")
        else:
            self.img_path = img_path  
            print("Selected image:", self.img_path) 
            
    def pushButton2F(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        augmentation_labels=[]
        if not folder_path:
            QMessageBox.warning(self, "操作錯誤提示", "請重新選擇資料夾")
        else:
            self.loadFiles = folder_path  
            print("Selected folder:", self.loadFiles)  
            
            for file_name in sorted(os.listdir(self.loadFiles)):
                if file_name.endswith(".png"):
                    self.files.append(os.path.join(self.loadFiles, file_name))
                    print(file_name)
                    label=os.path.splitext(file_name)[0]
                    print(label)
                    augmentation_labels.append(label)
        HW2_Q1.show_augmentation(self.files,augmentation_labels,self.transform)
        
    def pushButton3F(self):
        HW2_Q1.show_structure(self.model)

    def pushButton4F(self):
        HW2_Q1.show_loss_acc(self.acc,self.loss)
        
    def pushButton5F(self):
        HW2_Q1.show_inference(self)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Main_UI()
    window.show()
    sys.exit(app.exec_())