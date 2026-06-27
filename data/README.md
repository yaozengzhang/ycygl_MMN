# 数据集来源

本目录用于记录数据集公开来源和引用依据。

## Twitter

- Dataset: MediaEval 2015 Verifying Multimedia Use
- GitHub: https://github.com/MKLab-ITI/image-verification-corpus/tree/master/mediaeval2015
- Source repository: MKLab-ITI/image-verification-corpus
- Citation: Boididou, C., Papadopoulos, S., Zampoglou, M., Apostolidis, L., Papadopoulou, O., & Kompatsiaris, Y. (2018). Detection and visualization of misleading content on Twitter. International Journal of Multimedia Information Retrieval, 7(1), 71-86.
- 认证依据: 源仓库说明 `mediaeval2015` 文件夹是 MediaEval Workshop 2015 的 Verifying Multimedia Use task 数据版本。

## Weibo

- Dataset: Weibo multimodal rumor detection dataset
- GitHub: https://github.com/wangzhuang1911/Weibo-dataset
- Source repository: wangzhuang1911/Weibo-dataset
- Citation: Jin, Z., Cao, J., Guo, H., Zhang, Y., & Luo, J. (2017). Multimodal Fusion with Recurrent Neural Networks for Rumor Detection on Microblogs. ACM Multimedia 2017, 795-816.
- 认证依据: 源仓库说明这是 multimodal Weibo dataset，并要求引用上面的 ACM Multimedia 2017 论文。

## 复现数据放置

训练代码需要文本样本和原图图片。默认读取 `data/raw`，也可以通过环境变量 `YCYGL_DATA_ROOT` 指定数据根目录。

```text
data/raw/
├── weibo/
│   ├── text/
│   └── images/
└── twitter/
    ├── text/
    └── images/
```

完整图片和生成的 `.pt` 特征不作为普通 Git 文件提交。
