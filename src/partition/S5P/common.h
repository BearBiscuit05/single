#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <list>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <future>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdint>
#include <sys/types.h>
#include <fcntl.h>
#include <stdexcept>
#include <limits>
#include "omp.h"
#include <parallel_hashmap/phmap.h>
# define THREADNUM 8
# define NODENUM 3997962
# define EDGENUM 34681189