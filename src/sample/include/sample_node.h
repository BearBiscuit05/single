void launch_sample_full(int* outputSRC1,
                        int* outputDST1,
                 const int* graphEdge,
                 const int* boundList,
                 const int* trainNode,
                 int n,
                 const int gpuDeviceIndex);

void launch_sample_1hop(int* outputSRC1,
                        int* outputDST1, 
                        const int* graphEdge,
                        const int* boundList,
                        const int* trainNode,
                        int sampleNUM1,
                        int nodeNUM,
                        const int gpuDeviceIndex) ;

void launch_sample_2hop(
                        int* outputSRC1,
                        int* outputDST1,
                        int* outputSRC2,
                        int* outputDST2,
                        const int* graphEdge,
                        const int* boundList,
                        const int* trainNode,
                        int sampleNUM1,
                        int sampleNUM2,
                        int nodeNUM,
                        const int gpuDeviceIndex) ;

void launch_sample_3hop(int* outputSRC1,int* outputDST1,
                        int* outputSRC2,int* outputDST2,
                        int* outputSRC3,int* outputDST3,
                        const int* graphEdge,
                        const int* boundList,
                        const int* trainNode,
                        int sampleNUM1,int sampleNUM2,int sampleNUM3,
                        int nodeNUM,
                        const int gpuDeviceIndex) ;

void lanch_loading_halo(int* cacheData0,
                        int* cacheData1,
                        const int* edges,
                        const int* bound,
                        const int cacheData0Len,
                        const int cacheData1Len,
                        const int edgesLen,
                        const int boundLen,
                        const int graphEdgeNUM,
                        const int gpuDeviceIndex);

void lanch_loading_halo0(int* cacheData0,
                        int* cacheData1,
                        const int* edges,
                        const int cacheData0Len,
                        const int cacheData1Len,
                        const int edgesLen,
                        const int graphEdgeNUM,
                        const int gpuDeviceIndex);