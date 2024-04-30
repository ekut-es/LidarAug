
#include "../include/utils.hpp"
#include <algorithm>
#include <numeric>

template <template <typename...> class Container, typename T>
[[nodiscard]] Container<std::size_t> cpp_utils::argsort(const Container<T> &c,
                                                        bool descending) {

  Container<size_t> idx(c.size());
  std::iota(idx.begin(), idx.end(), 0);

  if (descending) {
    std::stable_sort(idx.begin(), idx.end(),
                     [&c](size_t i1, size_t i2) { return c[i1] > c[i2]; });

  } else {
    std::stable_sort(idx.begin(), idx.end(),
                     [&c](size_t i1, size_t i2) { return c[i1] < c[i2]; });
  }

  return idx;
}
