#include <iostream>
#include <math.h>
#include <assert.h>

// softmax: np.exp(x) / sum(np.exp(x))
void softmax(double* input, size_t size) {
  double sum = 0.0;
	for (int i = 0; i < size; ++i) {
		sum += exp(input[i]);
	}

	for (int i = 0; i < size; ++i) {
		input[i] = exp(input[i]) / sum;
	}
}

// softmax: np.exp(x-max) / sum(np.exp(x-max))
// https://slaystudy.com/implementation-of-softmax-activation-function-in-c-c/
void softmax2(double* input, size_t size) {
	assert(0 <= size <= sizeof(input) / sizeof(double));

	int i;
	double m, sum, constant;

	m = -INFINITY;
	for (i = 0; i < size; ++i) {
		if (m < input[i]) {
			m = input[i];
		}
	}

  // Subtracting the maximum from each value of the input array
  // ensures that the exponent doesn’t overflow.
	sum = 0.0;
	for (i = 0; i < size; ++i) {
		sum += exp(input[i] - m);
	}

	constant = m + log(sum);
	for (i = 0; i < size; ++i) {
		input[i] = exp(input[i] - constant);
	}
}

int main(int argc, char** argv) {
	double input[] = {1., 4.2, 0.6, 1.23, 4.3, 1.2, 2.5};
	int i, n = sizeof(input) / sizeof(double);

	printf("Input Array: ");
	for (i = 0; i < n; ++i)
		printf("%lf ", input[i]);
  printf("\n");

	softmax2(input, n);

	printf("Softmax Array: ");
	for (i = 0; i < n; ++i)
		printf("%lf ", input[i]);
  printf("\n");

  return 0;
}
