package tensor_test

import (
	"fmt"
	"log"
	"math/cmplx"

	"github.com/fumin/tensor"
)

func Example() {
	a := tensor.T3([][][]complex64{
		{{-6i, 5}, {-1 + 1i, -1 - 1i}, {4, -3 + 3i}},
		{{0, 0}, {0, 0}, {0, 0}},
	})

	// Slice and reshape.
	b := a.Slice([][2]int{{0, 1}, {0, 3}, {0, 2}})
	b = b.Reshape(3, 2)
	fmt.Println("Slice and reshape", b.ToSlice2())

	// Transpose and update.
	b = b.Transpose(1, 0)
	b = b.Slice([][2]int{{0, 2}, {1, 3}})
	b.SetAt([]int{0, 1}, -2-2i)
	fmt.Println("Transpose and update", b.ToSlice2())

	// Calculate eigenvalues of b.
	bufs := [3]*tensor.Dense{tensor.Zeros(1), tensor.Zeros(1), tensor.Zeros(1)}
	eigvals, eigvecs := tensor.Zeros(1), tensor.Zeros(1)
	if err := tensor.Eig(eigvals, eigvecs, b, bufs); err != nil {
		log.Fatalf("%v", err)
	}
	for i, vi := range []complex64{-3 + 1i, -1 + 3i} {
		if abs(eigvals.At(i)-vi) < 1e-6 {
			fmt.Printf("Eigenvalue %d: %v\n", i, vi)
		}
	}

	// Output:
	// Slice and reshape [[(0-6i) (5+0i)] [(-1+1i) (-1-1i)] [(4+0i) (-3+3i)]]
	// Transpose and update [[(-1+1i) (-2-2i)] [(-1-1i) (-3+3i)]]
	// Eigenvalue 0: (-3+1i)
	// Eigenvalue 1: (-1+3i)
}

func abs(x complex64) float64 {
	return cmplx.Abs(complex128(x))
}
