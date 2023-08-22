#include "bloomfilter.h"
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <set>

#define MAX_ITEMS 6000000      // 设置最大元素个数
#define ADD_ITEMS 1000      // 添加测试元素
#define P_ERROR 0.0001// 设置误差

struct edge
{
    int u;
    int v;
    int w;
};

// 中心点：3
struct edge edges[] = {
    {0,1,1},
    {1,2,1},
    {2,3,1},
    {3,4,1},
    {4,5,1},
    {4,6,1},
    {3,7,1},
    {3,8,1},
    {7,9,1},
    {8,10,1},
    {10,11,1},
    {9,12,1}
};

int main(int argc, char** argv)
{

    printf(" test bloomfilter\n");

    // 1. 定义BaseBloomFilter
    static BaseBloomFilter stBloomFilter = {0};

    // 2. 初始化stBloomFilter，调用时传入hash种子，存储容量，以及允许的误判率
    InitBloomFilter(&stBloomFilter, 0, MAX_ITEMS, P_ERROR);

    const int startNode = 3;

    printf("start adding nodes...");
    if(0 == BloomFilter_Add(&stBloomFilter,(const void*)&startNode,sizeof(int))){
        printf("add node %d success\n",startNode);
    }else{
        printf("add node %d fail\n",startNode);
    }

    int weight[2] = {50,20};

    for(int i = 0;i < 2;i++)
    {
        printf("epoch: %d\n",i+1);
        std::set<int> nodes;
        int cnt = 0;
        for(edge& e: edges)
        {
            int ret = BloomFilter_CheckEdge(&stBloomFilter,e.u,e.v);
            if(ret == 0)
            {
                printf("edge[%2d->%2d] isn't be selected.\n",e.u,e.v);
            }
            else if(ret == 1)
            {
                printf("edge[%2d->%2d] is selected.\n",e.u,e.v);
                nodes.insert(e.u);
                e.w += weight[i];
            }
            else if(ret == 2)
            {
                printf("edge[%2d->%2d] is selected.\n",e.u,e.v);
                nodes.insert(e.v);
                e.w += weight[i];
            }
            else
            {
                printf("edge[%2d->%2d] has been selected before.\n",e.u,e.v);
            }
        }

        printf("nodes cache to be added:");
        for(int nid:nodes)
        {
            printf("%d ",nid);
        }
        printf("\n");

        if(0 == BloomFilter_AddNodes(&stBloomFilter,nodes)){
            printf("add nodes success\n");
        }else{
            printf("add nodes failed\n");
        }
    }
    
    for(edge&e : edges)
    {
        printf("edge[%2d->%2d] weight = %d\n",e.u,e.v,e.w);
    }

    // 3. 向BloomFilter中新增数值
    // char url[128] = {0};
    // for(int i = 0; i < ADD_ITEMS; i++){
    //     sprintf(url, "https://blog.csdn.net/qq_41453285/%d.html", i);
    //     if(0 == BloomFilter_Add(&stBloomFilter, (const void*)url, strlen(url))){
    //         // printf("add %s success", url);
    //     }else{
    //         printf("add %s failed", url);
    //     }
    //     memset(url, 0, sizeof(url));
    // }

    // 4. check url exist or not
    // char* str = "https://blog.csdn.net/qq_41453285/0.html";
    // if (0 == BloomFilter_Check(&stBloomFilter, (const void*)str, strlen(str)) ){
    //     printf("https://blog.csdn.net/qq_41453285/0.html exist\n");
    // }

    // char* str2 = "https://blog.csdn.net/qq_41453285/10001.html";
    // if (0 != BloomFilter_Check(&stBloomFilter, (const void*)str2, strlen(str2)) ){
    //       printf("https://blog.csdn.net/qq_41453285/10001.html not exist\n");
    // }

    // 5. free bloomfilter
    FreeBloomFilter(&stBloomFilter);
    getchar();
    return 0;
}