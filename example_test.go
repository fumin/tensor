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

func ExampleProduct() {
	a := tensor.T3([][][]complex64{
		{{0, 1 + 1i}, {2, 3}, {4, 5}},
		{{6, 7}, {8, 9}, {10, 11}},
		{{12, 13}, {14, 15}, {16, 17}},
	})
	b := tensor.T3([][][]complex64{
		{{0, 1, 2}, {3, 4, 5}},
		{{6, 7, 8}, {9, 10, 11}},
	})
	out := tensor.Zeros(1)
	tensor.Product(out, a, b, [][2]int{{1, 2}, {2, 1}})
	fmt.Println("a * b =", out.ToSlice2())

	// Output:
	// a * b = [[(50+3i) (140+9i)] [(140+0i) (446+0i)] [(230+0i) (752+0i)]]
}

func ExampleProduct_kronecker() {
	a := tensor.T2([][]complex64{{1, 2}, {3, 4}})
	b := tensor.T2([][]complex64{{5, 6}, {7, 8}})

	ops := []*tensor.Dense{a, b, b, a, a}

	// Compute tensor product.
	out, buf := tensor.Zeros(1), tensor.Zeros(1)
	out.Reset(ops[0].Shape()...).Set(nil, ops[0])
	n := ops[0].Shape()[0]
	for _, op := range ops[1:] {
		tensor.Product(buf, out, op, nil)
		out.Reset(buf.Shape()...).Set(nil, buf)
		n *= op.Shape()[0]
	}
	// Reshape to matrix.
	out = out.Transpose(0, 2, 4, 6, 8, 1, 3, 5, 7, 9)
	buf.Reset(out.Shape()...).Set(nil, out)
	out = buf.Reshape(n, n)

	// Print the result.
	fmt.Printf("Kronecker product (a * b * b * a * a): %v...\n", out.ToSlice2()[0][:5])

	// Output:
	// Kronecker product (a * b * b * a * a): [(25+0i) (50+0i) (50+0i) (100+0i) (30+0i)]...
}

func abs(x complex64) float64 {
	return cmplx.Abs(complex128(x))
}
