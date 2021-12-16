#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

double my_exp_func(double num) {
  return exp(num);
}

// softmax: np.exp(x) / sum(np.exp(x))
template <typename T>
void softmax(const std::vector<T> &v, std::vector<T> &s){
    double sum = 0.0;
    std::transform(v.begin(), v.end(), s.begin(), my_exp_func);
    sum = std::accumulate(s.begin(), s.end(), sum);
    for (size_t i = 0; i < s.size(); ++i)
      s.at(i) /= sum;
}

int main() {
    double a[] = {1., 4.2, 0.6, 1.23, 4.3, 1.2, 2.5};
    std::vector<double> v_a(a, a + (sizeof(a) / sizeof(a[0]))), v_b(v_a);
    std::vector<double>::const_iterator it = v_a.begin();

    std::cout << "Input Array: ";
    for (; it != v_a.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout<<std::endl;

    softmax(v_a, v_b);

    std::cout << "Softmax Array: ";
    it = v_b.begin();
    for (; it != v_b.end(); ++it) {
        std::cout << *it <<" ";
    }
    std::cout<<std::endl;

    return 0;
}
