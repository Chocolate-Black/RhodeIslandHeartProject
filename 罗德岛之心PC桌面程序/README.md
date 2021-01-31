## 源代码说明

- main.py:主程序
- comb_box_data.py:干员信息，用于筛选框
- main_ui.py,main_ui.ui:UI设计
- lgb_model.py:加载模型，用于预测
- main.spec:打包配置文件

目前只做了windows部分，如果想弄mac版的可以在本目录下运行：

```
pyinstaller -D main.spec
```

程序会保存在dist里，**注意：如果想要正确运行，请把程序剪切到dist外面的文件夹里，否则相对路径不对无法运行**
