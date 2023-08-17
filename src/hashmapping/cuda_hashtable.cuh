#include <cuda_runtime.h>
#include <cassert>
#include <stdint.h>
#include <cub/cub.cuh>
#include <iostream>
#define CUDA_CALL(func)                                      \
  {                                                          \
    cudaError_t e = (func);                                  \
    if(e!=cudaSuccess) { \
		std::cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(e) << " (" << e << ")" << std::endl; \
		exit(0); \
	 }\
  }

template <typename>
class OrderedHashTable;

template <typename IdType>
class DeviceOrderedHashTable {
  public:
    struct Mapping {
      IdType key;
      IdType local;
      int64_t index;
    };

    typedef const Mapping* ConstIterator;

    DeviceOrderedHashTable(const DeviceOrderedHashTable& other) = default;
    DeviceOrderedHashTable& operator=(const DeviceOrderedHashTable& other) =
      default;
  
    inline __device__ ConstIterator Search(const IdType id) const {
      const IdType pos = SearchForPosition(id);
      return &table_[pos];
    }

    inline __device__ bool Contains(const IdType id) const {
      IdType pos = Hash(id);
      IdType delta = 1;
      while (table_[pos].key != kEmptyKey) {
        if (table_[pos].key == id) {
          return true;
        }
        pos = Hash(pos + delta);
        delta += 1;
      }
      return false;
    }

  // protected:
    static constexpr IdType kEmptyKey = static_cast<IdType>(-1);
    const Mapping* table_;
    size_t size_;

    explicit DeviceOrderedHashTable(const Mapping* table, 
                                    size_t size);

    inline __device__ IdType SearchForPosition(const IdType id) const {
      IdType pos = Hash(id);
      IdType delta = 1;
      while (table_[pos].key != id) {
        assert(table_[pos].key != kEmptyKey);
        pos = Hash(pos + delta);
        delta += 1;
      }
      assert(pos < size_);
      return pos;
    }
    inline __device__ size_t Hash(const IdType id) const { return id % size_; }
    friend class OrderedHashTable<IdType>;
};

template <typename IdType>
class OrderedHashTable { 
public:
  static constexpr int kDefaultScale = 2;
  using Mapping = typename DeviceOrderedHashTable<IdType>::Mapping;

  OrderedHashTable(
      const size_t size,
      const int scale = kDefaultScale);
  
  ~OrderedHashTable();

  OrderedHashTable(const OrderedHashTable& other) = delete;
  OrderedHashTable& operator=(const OrderedHashTable& other) = delete;

  void FillWithDuplicates(
      IdType* input, size_t num_input, IdType* unique,
      int64_t* num_unique);
    
  void FillWithUnique(
      const IdType* const input, const size_t num_input);
  
  DeviceOrderedHashTable<IdType> DeviceHandle() const;

  // private:
  Mapping* table_;
  size_t size_;
  
};

