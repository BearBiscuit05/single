# 测试文件

## 已知错误：
1. valID.bin和testID.bin错误地保存了原始特征中train_mask的数据，位置在predata/graph2bin.py的函数gen_ids_file三个torch.save
