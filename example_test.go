package tensor_test

import (
	"fmt"
	"log"
	"math"
	"math/cmplx"
	"strconv"

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

func ExampleDense_Set() {
	a := tensor.T2([][]complex64{{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}})
	b := tensor.T2([][]complex64{{1, 2, 3}, {4, 5, 6}})

	a.Set([]int{3, 2}, b)
	fmt.Printf("a.Set({3, 2}, b):\n")
	printInt(a)

	a.Set([]int{0, -4}, b)
	fmt.Printf("a.Set({0, -4}, b):\n")
	printInt(a)

	a.Set(nil, b)
	fmt.Printf("a.Set(nil, b):\n")
	printInt(a)

	// Output:
	// a.Set({3, 2}, b):
	// 0 0 0 0 0 0
	// 0 0 0 0 0 0
	// 0 0 0 0 0 0
	// 0 0 1 2 3 0
	// 0 0 4 5 6 0
	// 0 0 0 0 0 0
	//
	// a.Set({0, -4}, b):
	// 0 0 1 2 3 0
	// 0 0 4 5 6 0
	// 0 0 0 0 0 0
	// 0 0 1 2 3 0
	// 0 0 4 5 6 0
	// 0 0 0 0 0 0
	//
	// a.Set(nil, b):
	// 1 2 3 2 3 0
	// 4 5 6 5 6 0
	// 0 0 0 0 0 0
	// 0 0 1 2 3 0
	// 0 0 4 5 6 0
	// 0 0 0 0 0 0
}

func ExampleDense_Reshape() {
	a := tensor.T1([]complex64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})

	fmt.Printf("a.Reshape(2, 6):\n")
	printInt(a.Reshape(2, 6))

	fmt.Printf("a.Reshape(3, -1):\n")
	printInt(a.Reshape(3, -1))

	// Output:
	// a.Reshape(2, 6):
	// 0 1 2 3 4 5
	// 6 7 8 9 10 11
	//
	// a.Reshape(3, -1):
	// 0 1 2 3
	// 4 5 6 7
	// 8 9 10 11
	//
}

func ExampleDense_Slice() {
	a := tensor.T2([][]complex64{{0, 1, 2, 3, 4, 5}, {6, 7, 8, 9, 10, 11}, {12, 13, 14, 15, 16, 17}})

	b := a.Slice([][2]int{{1, 3}, {2, 5}})
	fmt.Printf("a[1:3, 2:5] = %v\n", b.ToSlice2())

	b = a.Slice([][2]int{{1, 3}, {2, -2}})
	fmt.Printf("a[1:3, 2:-2] = %v\n", b.ToSlice2())

	b = a.Slice([][2]int{{1, 0}, {2, -2}})
	fmt.Printf("a[1:, 2:-2] = %v\n", b.ToSlice2())

	b = a.Slice([][2]int{{}, {-4, -2}})
	fmt.Printf("a[:, -4:-2] = %v\n", b.ToSlice2())

	// Output:
	// a[1:3, 2:5] = [[(8+0i) (9+0i) (10+0i)] [(14+0i) (15+0i) (16+0i)]]
	// a[1:3, 2:-2] = [[(8+0i) (9+0i)] [(14+0i) (15+0i)]]
	// a[1:, 2:-2] = [[(8+0i) (9+0i)] [(14+0i) (15+0i)]]
	// a[:, -4:-2] = [[(2+0i) (3+0i)] [(8+0i) (9+0i)] [(14+0i) (15+0i)]]
}

func ExampleSVD() {
	// This example shows how we can use the SVD to compute
	// the kernel, image, and cokernel of a linear map.
	a := tensor.T2([][]complex64{
		{1, 2, 3, 2},
		{1, 1, 2, 2},
		{1, 3, 4, 2},
	})

	// Compute the SVD.
	bufs := [3]*tensor.Dense{tensor.Zeros(1), tensor.Zeros(1), tensor.Zeros(1)}
	u, v := tensor.Zeros(1), tensor.Zeros(1)
	s, _ := tensor.SVD(u, v, a, bufs, tensor.SVDOptions{Full: true})

	// Collect the kernel, image, and cokernel.
	const tol = 1e-5
	kernel := make([][]float32, 0)
	image := make([][]float32, 0)
	cokernel := make([][]float32, 0)
	m, n := s.Shape()[0], s.Shape()[1]
	for i := range u.Shape()[1] {
		ui := u.Slice([][2]int{{}, {i, i + 1}})
		if !(i < m && i < n) || real(s.At(i, i)) < tol {
			cokernel = append(cokernel, realSlice(ui))
		} else {
			image = append(image, realSlice(ui))
		}
	}
	for i := range v.Shape()[1] {
		vi := v.Slice([][2]int{{}, {i, i + 1}})
		if !(i < m && i < n) || real(s.At(i, i)) < tol {
			kernel = append(kernel, realSlice(vi))
		}
	}

	fmt.Println("kernel:")
	printFloats(kernel, 3)
	fmt.Println("image:")
	printFloats(image, 3)
	fmt.Println("cokernel:")
	printFloats(cokernel, 3)

	// Output:
	// kernel:
	// [0.902, 0.055, -0.055, -0.424]
	// [0.061, 0.672, -0.672, 0.305]
	//
	// image:
	// [0.562, 0.403, 0.722]
	// [0.130, 0.819, -0.558]
	//
	// cokernel:
	// [0.816, -0.408, -0.408]
}

func printInt(t *tensor.Dense) {
	m := t.Shape()[0]
	for i := range m {
		row := intSlice(t.Slice([][2]int{{i, i + 1}, {}}))
		for _, v := range row[:len(row)-1] {
			fmt.Printf("%d ", v)
		}
		fmt.Printf("%d\n", row[len(row)-1])
	}
	fmt.Printf("\n")
}

func printFloats(xs [][]float32, prec int) {
	for _, x := range xs {
		fmt.Printf("[")
		for i := range len(x) - 1 {
			fmt.Printf("%s, ", strconv.FormatFloat(float64(x[i]), 'f', prec, 32))
		}
		fmt.Printf("%s]\n", strconv.FormatFloat(float64(x[len(x)-1]), 'f', prec, 32))
	}
	fmt.Printf("\n")
}

func intSlice(x *tensor.Dense) []int {
	y := realSlice(x)
	z := make([]int, len(y))
	for i := range y {
		z[i] = int(math.Round(float64(y[i])))
	}
	return z
}

func realSlice(x *tensor.Dense) []float32 {
	flat := tensor.Zeros(x.Shape()...).Set(nil, x)
	flat = flat.Reshape(-1)
	y := make([]float32, flat.Shape()[0])
	for i := range flat.Shape()[0] {
		y[i] = real(flat.At(i))
	}
	return y
}

func abs(x complex64) float64 {
	return cmplx.Abs(complex128(x))
}
