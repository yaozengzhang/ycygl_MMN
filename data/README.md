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

训练代码需要文本样本和原图图片。默认路径在 `gcn_ycygl_pipeline\config.py` 中配置；如果复现环境路径不同，修改该文件中的 `DATASETS` 即可。

```text
sample_dir: 文本样本目录
image_dir: 原图图片目录
```

完整图片和生成的 `.pt` 特征通常体积较大，建议不要直接作为普通 Git 文件提交。
