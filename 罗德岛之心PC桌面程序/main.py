import sys
import main_ui
from PyQt5.QtWidgets import QApplication,QMainWindow,\
    QHBoxLayout,QWidget,QComboBox,QLabel,\
    QListWidgetItem,QMessageBox,QFrame
from PyQt5 import QtCore, QtGui
import comb_box_data
import lgb_model

class MainWindow(QMainWindow,main_ui.Ui_mainWindow):
    def __init__(self,parent=None):
        super(MainWindow,self).__init__(parent)
        self.setupUi(self)
        self.setFixedSize(self.width(), self.height())
        self.setWindowTitle("罗德岛之心精二box推荐器")
        self.setWindowIcon(QtGui.QIcon('./icon/logo.ico'))
        # 筛选部分
        self.comboBox.clear()
        self.comboBox_2.clear()
        self.comboBox_3.clear()
        self.comboBox.addItem('全部星级',-1)
        self.comboBox_2.addItem('全部职业',-1)
        self.level_combo_group = {}
        for k,v in comb_box_data.star.items():
            self.comboBox.addItem(v,k)
        for k,v in comb_box_data.agent_class.items():
            self.comboBox_2.addItem(v,k)
        for k in comb_box_data.agents_dict.keys():
            self.comboBox_3.addItem(k)

        self.comboBox.currentIndexChanged.connect(self.selection_changed)
        self.comboBox_2.currentIndexChanged.connect(self.selection_changed)
        self.pushButton.clicked.connect(self.add_agent)
        self.pushButton_2.clicked.connect(self.get_agent_score)
        self.pushButton_3.clicked.connect(self.delete_agent)

        # 存储box值，用于模型输入
        self.agent_box = []
        # 加载模型
        self.my_model = lgb_model.Model()

    def messageDialog(self,text):
        # 核心功能代码就两行，可以加到需要的地方
        msg_box = QMessageBox(QMessageBox.Warning, '警告', text)
        msg_box.exec_()

    def selection_changed(self):
        self.comboBox_3.clear()
        star_id = self.comboBox.currentIndex()
        agent_class_id = self.comboBox_2.currentIndex()
        star = self.comboBox.itemData(star_id)
        agent_class = self.comboBox_2.itemData(agent_class_id)
        for k,v in comb_box_data.agents_dict.items():
            # 判断是否符合筛选条件
            if (v['star'] == star or star == -1) and (v['class'] == agent_class or agent_class == -1):
                self.comboBox_3.addItem(k)

    # 添加干员
    def add_agent(self):
        # 获取干员信息
        agent_name_id = self.comboBox_3.currentIndex()
        agent_name= self.comboBox_3.itemText(agent_name_id)

        if agent_name in self.level_combo_group.keys():
            self.messageDialog("干员已添加！")
        else:
            star_value = comb_box_data.agents_dict[agent_name]['star']
            agent_class_value = comb_box_data.agents_dict[agent_name]['class']
            file_path = comb_box_data.agents_dict[agent_name]['profile']
            star = comb_box_data.star[star_value]
            agent_class = comb_box_data.agent_class[agent_class_value]


            # 窗口设置
            wight = QWidget()
            row_layout = QHBoxLayout()
            # 等级调整comb
            comb = QComboBox()
            comb.setObjectName(agent_name)
            for k, v in comb_box_data.level.items():
                comb.addItem(v, k)
            self.level_combo_group[agent_name] = comb
            comb.currentIndexChanged.connect(lambda: self.change_agent_level(agent_name))
            # 加入图片
            profile = QLabel()

            profile.setFixedSize(50,50)
            png = QtGui.QPixmap(file_path)
            profile.setPixmap(png)
            profile.setAlignment(QtCore.Qt.AlignVCenter)
            profile.setScaledContents(True)

            # 设置文本
            row_layout.addWidget(profile)
            row_layout.addWidget(QLabel(agent_name))
            line = QFrame(self.layoutWidget)
            line.setLineWidth(1)
            line.setFrameShape(QFrame.VLine)
            line.setFrameShadow(QFrame.Sunken)
            row_layout.addWidget(line)
            row_layout.addWidget(QLabel('星级：'+star))
            row_layout.addWidget(QLabel('职业：'+agent_class))
            row_layout.addWidget(comb)
            # row_layout.addWidget(delete_button)
            wight.setLayout(row_layout)
            # 添加项目
            item = QListWidgetItem()
            item.setSizeHint(QtCore.QSize(200, 70))
            self.listWidget_2.addItem(item)
            self.listWidget_2.setItemWidget(item, wight)
            dic = {'name':agent_name,'level':0}
            self.agent_box.append(dic)
            # print('add is ok!')
            # print(self.agent_box)

    def delete_agent(self):
        if self.listWidget_2.count() == 0:
            self.messageDialog("没有可删除的干员！")
        else:
            try:
                item = self.listWidget_2.currentItem()
                index = self.listWidget_2.currentIndex()
                self.listWidget_2.takeItem(self.listWidget_2.row(item))
                row = index.row()
                if row != -1:
                    dic = self.agent_box.pop(row)
                    self.level_combo_group.pop(dic['name'])
                    # print("delete is ok!")
                    # print(self.agent_box)
                else:
                    self.messageDialog("请先选中干员！")
            except:
                self.messageDialog("请先选中干员！")


    def change_agent_level(self,name):
        comb = self.findChild(QComboBox,name)
        level_id = comb.currentIndex()
        level = comb.itemData(level_id)
        for dic in self.agent_box:
            if dic['name'] == name:
                dic['level'] = level
                print(self.agent_box)
                break

    def get_agent_score(self):
        if len(self.agent_box) == 0:
            self.messageDialog("请先添加干员！")
        else:
            result = self.my_model.predict(self.agent_box)
            # print(result)
            self.rank(result)

    def rank(self,data):
        self.listWidget.clear()
        for agent in data['predictions']:
            self.listWidget.addItem(agent['name']+'   '+'score=%.3f' % agent['score'])
        # print("rank is ok!")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())