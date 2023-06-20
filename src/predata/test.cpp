#include <iostream>
#include <vector>

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};

    std::cout << "Original vector: ";
    for (const auto& num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    std::cout << "Size before resize: " << numbers.size() << std::endl;

    // 调整向量的大小为 8
    numbers.resize(8);

    std::cout << "Vector after resize (larger size): ";
    for (const auto& num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    std::cout << "Size after resize (larger size): " << numbers.size() << std::endl;

    // 调整向量的大小为 3
    numbers.resize(3);

    std::cout << "Vector after resize (smaller size): ";
    for (const auto& num : numbers) {
        std::cout << num << " ";
    }
    std::cout << std::endl;

    std::cout << "Size after resize (smaller size): " << numbers.size() << std::endl;

    return 0;
}
