#include <cassert>
#include "cuda_hashtable.h"

#define NUM 64

inline __device__ int64_t
AtomicCAS(int64_t* const address, const int64_t compare, const int64_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = unsigned long long int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicCAS(
      reinterpret_cast<Type*>(address), static_cast<Type>(compare),
      static_cast<Type>(val));
}

inline __device__ int32_t
AtomicCAS(int32_t* const address, const int32_t compare, const int32_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = int;  // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicCAS(
      reinterpret_cast<Type*>(address), static_cast<Type>(compare),
      static_cast<Type>(val));
}

void *AllocDataSpace(size_t nbytes) {
    void *ret = nullptr;
    CUDA_CALL(cudaMalloc(&ret, nbytes));
    return ret;
}

void FreeDataSpace(void *ret) {
  CUDA_CALL(cudaFree(ret));
}

constexpr static const int BLOCK_SIZE = 256;
constexpr static const size_t TILE_SIZE = 1024;


// using namespace cuda;
template <typename IdType>
class MutableDeviceOrderedHashTable : public DeviceOrderedHashTable<IdType> {
public:
  typedef typename DeviceOrderedHashTable<IdType>::Mapping* Iterator;
  static constexpr IdType kEmptyKey = DeviceOrderedHashTable<IdType>::kEmptyKey;

  explicit MutableDeviceOrderedHashTable(
      OrderedHashTable<IdType>* const hostTable)
      : DeviceOrderedHashTable<IdType>(hostTable->DeviceHandle()) {
        // printf("create success hashTable\n");
      }

  inline __device__ Iterator Search(const IdType id) {
    const IdType pos = SearchForPosition(id);
    return GetMutable(pos);
  }

  inline __device__ bool AttemptInsertAt(
      const size_t pos, const IdType id, const size_t index) {
    const IdType key = AtomicCAS(&GetMutable(pos)->key, kEmptyKey, id);
    if (key == kEmptyKey || key == id) {
      atomicMin(
          reinterpret_cast<unsigned long long*>(  // NOLINT
              &GetMutable(pos)->index),
          static_cast<unsigned long long>(index));  // NOLINT
      return true;
    } else {
      return false;
    }
  }
  // size_t real_size=0;
  inline __device__ Iterator Insert(const IdType id, const size_t index) {
    size_t pos = Hash(id);
    // printf("get insert id : %d \n",id);
    IdType delta = 1;
    while (!AttemptInsertAt(pos, id, index)) {
      pos = Hash(pos + delta);
      delta += 1;
    }
    
    return GetMutable(pos);
  }

  inline __device__ Iterator GetMutable(const size_t pos) {
    assert(pos < this->size_);
    return const_cast<Iterator>(this->table_ + pos);
  }
};


// 计算哈希表大小
size_t TableSize(const size_t num, const int scale) {
  const size_t next_pow2 = 1 << static_cast<size_t>(1 + std::log2(num >> 1));
  return next_pow2 << scale;
}

template <typename IdType>
struct BlockPrefixCallbackOp {
  IdType running_total_;

  __device__ BlockPrefixCallbackOp(const IdType running_total)
      : running_total_(running_total) {}

  __device__ IdType operator()(const IdType block_aggregate) {
    const IdType old_prefix = running_total_;
    running_total_ += block_aggregate;
    return old_prefix;
  }
};

template <typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_hashmap_duplicates(
    const IdType* const items, const int64_t num_items,
    MutableDeviceOrderedHashTable<IdType> table,int test) {
  
  assert(BLOCK_SIZE == blockDim.x);

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      // printf("gen items: %d \n",items[index]);
      table.Insert(items[index], index);
    }
  }
  
  
}

template <typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void generate_hashmap_unique(
    const IdType* const items, const int64_t num_items,
    MutableDeviceOrderedHashTable<IdType> table) {
  assert(BLOCK_SIZE == blockDim.x);

  using Iterator = typename MutableDeviceOrderedHashTable<IdType>::Iterator;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      const Iterator pos = table.Insert(items[index], index);
      pos->local = static_cast<IdType>(index);
    }
  }
}


template <typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void count_hashmap(
    const IdType* items, const size_t num_items,
    DeviceOrderedHashTable<IdType> table, IdType* const num_unique) {
  assert(BLOCK_SIZE == blockDim.x);
  using BlockReduce = typename cub::BlockReduce<IdType, BLOCK_SIZE>;
  using Mapping = typename DeviceOrderedHashTable<IdType>::Mapping;

  const size_t block_start = TILE_SIZE * blockIdx.x;
  const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

  IdType count = 0;

#pragma unroll
  for (size_t index = threadIdx.x + block_start; index < block_end;
       index += BLOCK_SIZE) {
    if (index < num_items) {
      const Mapping& mapping = *table.Search(items[index]);
      if (mapping.index == index) {
        ++count;
      }
    }
  }

  __shared__ typename BlockReduce::TempStorage temp_space;

  count = BlockReduce(temp_space).Sum(count);

  if (threadIdx.x == 0) {
    num_unique[blockIdx.x] = count;
    // printf("num_unique[blockIdx.x] %d \n",num_unique[blockIdx.x]);
    if (blockIdx.x == 0) {
      num_unique[gridDim.x] = 0;
    }
  }
}

template <typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void compact_hashmap(
    const IdType* const items, const size_t num_items,
    MutableDeviceOrderedHashTable<IdType> table,
    const IdType* const num_items_prefix, IdType* const unique_items,
    int64_t* num_unique_items) {
  assert(BLOCK_SIZE == blockDim.x);
  
  using FlagType = uint16_t;
  using BlockScan = typename cub::BlockScan<FlagType, BLOCK_SIZE>;
  using Mapping = typename DeviceOrderedHashTable<IdType>::Mapping;
  
  constexpr const int32_t VALS_PER_THREAD = TILE_SIZE / BLOCK_SIZE;
  __shared__ typename BlockScan::TempStorage temp_space;
  const IdType offset = num_items_prefix[blockIdx.x];
  BlockPrefixCallbackOp<FlagType> prefix_op(0);
  
  for (int32_t i = 0; i < VALS_PER_THREAD; ++i) {
    const IdType index = threadIdx.x + i * BLOCK_SIZE + blockIdx.x * TILE_SIZE;
    
    FlagType flag;
    Mapping* kv;
    if (index < num_items) {
      kv = table.Search(items[index]);
      // printf("kv->index : %d \n",kv->index);
      flag = kv->index == index;
    } else {
      flag = 0;
    }
    if (!flag) {
      kv = nullptr;
    }
    BlockScan(temp_space).ExclusiveSum(flag, flag, prefix_op);
    __syncthreads();

    if (kv) {
      const IdType pos = offset + flag;
      kv->local = pos;
      unique_items[pos] = items[index];
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // printf("num_unique_items[0] : %d \n",num_unique_items[0]);
    *num_unique_items = num_items_prefix[gridDim.x];
    // num_unique_items[0] = 6;
    // printf("num_unique_items[0] : %d \n",num_unique_items[0]);
  }
}


template <typename IdType>
DeviceOrderedHashTable<IdType>::DeviceOrderedHashTable(
    const Mapping* const table, const size_t size)
    : table_(table), size_(size) {}

template <typename IdType>
DeviceOrderedHashTable<IdType> OrderedHashTable<IdType>::DeviceHandle() const {
  return DeviceOrderedHashTable<IdType>(table_, size_);
}

template <typename IdType>
OrderedHashTable<IdType>::OrderedHashTable(
    const size_t size, const int scale)
    : table_(nullptr), size_(TableSize(size, scale)) {
  table_ = static_cast<Mapping*>(
      AllocDataSpace(sizeof(Mapping) * size_));

  CUDA_CALL(cudaMemsetAsync(
      table_, DeviceOrderedHashTable<IdType>::kEmptyKey,
      sizeof(Mapping) * size_));
}

template <typename IdType>
OrderedHashTable<IdType>::~OrderedHashTable() {
  FreeDataSpace(table_);
}

template <typename IdType>
void OrderedHashTable<IdType>::FillWithDuplicates(
    IdType* input, size_t num_input, IdType* unique,
    int64_t* num_unique) {
  const int64_t num_tiles = (num_input + TILE_SIZE - 1) / TILE_SIZE;
  const dim3 grid(num_tiles);
  const dim3 block(BLOCK_SIZE);

  IdType *dev_array;
  IdType *dev_uni_array;
  size_t dev_num=NUM;
  IdType test=-1;
  cudaMalloc(&dev_array, sizeof(IdType)*NUM);
	cudaMemcpy(dev_array, input, sizeof(IdType)*NUM, cudaMemcpyHostToDevice);
  cudaMemcpy(input, dev_array, sizeof(IdType)*NUM, cudaMemcpyDeviceToHost);
  // for(int i = 0 ; i < 8 ; i++){
  //   std::cout << "TESTING :" << input[i] << std::endl;}
    
  cudaMalloc(&dev_uni_array, sizeof(IdType)*NUM);
	cudaMemcpy(&dev_uni_array, unique, sizeof(IdType)*NUM, cudaMemcpyHostToDevice);
  
  auto device_table = MutableDeviceOrderedHashTable<IdType>(this);
  // std::cout << "device_table size :"<< device_table.size_ << std::endl;
  
  generate_hashmap_duplicates<IdType, BLOCK_SIZE, TILE_SIZE> 
  <<<grid, block>>>(dev_array, dev_num, device_table,test);
  CUDA_CALL(cudaDeviceSynchronize());
  IdType* item_prefix = static_cast<IdType*>(AllocDataSpace(sizeof(IdType) * (num_input + 1)));
  
  std::vector<IdType> dev_item(num_input + 1,0);
	cudaMemcpy(item_prefix, dev_item.data(), sizeof(IdType)*NUM, cudaMemcpyHostToDevice);
  
  
  count_hashmap<IdType, BLOCK_SIZE, TILE_SIZE>
  <<<grid, block>>>(dev_array, num_input, device_table, item_prefix);
  CUDA_CALL(cudaDeviceSynchronize());
  // cudaMemcpy(dev_item.data(), item_prefix, sizeof(IdType)*NUM, cudaMemcpyDeviceToHost);

  size_t workspace_bytes=0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    nullptr, workspace_bytes, static_cast<IdType*>(nullptr),
    static_cast<IdType*>(nullptr), grid.x + 1));
  CUDA_CALL(cudaDeviceSynchronize());
  printf("workspace_bytes : %zu \n",workspace_bytes);
  void* workspace = AllocDataSpace(workspace_bytes);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
    workspace, workspace_bytes, item_prefix, item_prefix, grid.x + 1));
  CUDA_CALL(cudaDeviceSynchronize());
  FreeDataSpace(workspace);
  cudaMemcpy(dev_item.data(), item_prefix, sizeof(int64_t)*NUM, cudaMemcpyDeviceToHost);

  int64_t* dev_num_unique;
  cudaMalloc(&dev_num_unique, sizeof(int64_t));
	cudaMemcpy(dev_num_unique, num_unique, sizeof(int64_t), cudaMemcpyHostToDevice);
  compact_hashmap<IdType, BLOCK_SIZE, TILE_SIZE><<<grid, block>>>
  (dev_array, num_input, device_table, item_prefix, dev_uni_array, dev_num_unique);
  CUDA_CALL(cudaDeviceSynchronize());
  FreeDataSpace(item_prefix);
  cudaMemcpy(num_unique, dev_num_unique, sizeof(int64_t), cudaMemcpyDeviceToHost);
  printf("num_unique : %zu \n",num_unique[0]);
}

template <typename IdType>
void OrderedHashTable<IdType>::FillWithUnique(
    const IdType* const input, const size_t num_input) {
  const int64_t num_tiles = (num_input + TILE_SIZE - 1) / TILE_SIZE;

  const dim3 grid(num_tiles);
  const dim3 block(BLOCK_SIZE);

  auto device_table = MutableDeviceOrderedHashTable<IdType>(this);

  generate_hashmap_unique<IdType, BLOCK_SIZE, TILE_SIZE>
  <<<grid, block>>>(input, num_input, device_table);
}

template class OrderedHashTable<int32_t>;
template class OrderedHashTable<int64_t>;

int main(){
  OrderedHashTable<int64_t> table(80);
  std::vector<int64_t> myVector = {2, 3, 4, 2, 7, 8};
  myVector.resize(NUM,0);
  std::vector<int64_t> myUnique(NUM,0);
  
  int64_t inputNUM = NUM;
  int64_t unique = 0;
  table.FillWithDuplicates(myVector.data(),inputNUM,myUnique.data(),&unique);
  std::cout << "uniqueNUM :"<< unique << std::endl;
  return 0;
}