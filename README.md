dgl数据结构介绍
模版最初进行定义,在后续采样过程中,每次采样的图结构并不一样，
因此对于每次的采样图，生成一个对应边掩码(edge id)，用于指出
原来的采样图哪部分结构需要调整，将模版存在，但是采样中不存在
的，全部修改为节点0的循环loop