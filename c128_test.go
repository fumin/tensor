package tensor

import (
	"fmt"
	"testing"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/cblas128"
)

func TestGemm(t *testing.T) {
	t.Parallel()
	tests := []struct {
		ts []*Dense
		p  [][]complex64
	}{
		{
			ts: []*Dense{
				T2([][]complex64{{1 - 1i, 2}, {3, 4i}}),
				T2([][]complex64{{-1i, -2i}, {-3i, -4i}}),
				T2([][]complex64{{2 + 3i, -2}, {-3 + 1i, -4}}),
			},
			p: [][]complex64{{35 + 11i, 10 + 54i}, {-9 + 64i, -88 + 30i}},
		},
	}
	for i, test := range tests {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			t.Parallel()
			p := gemm(test.ts...)
			if err := equal2(p, test.p, epsilon); err != nil {
				t.Fatalf("%+v", err)
			}
		})
	}
}

func to128(t *Dense) cblas128.General {
	g := cblas128.General{Rows: t.Shape()[0], Cols: t.Shape()[1]}
	g.Stride = g.Cols
	g.Data = make([]complex128, 0, g.Rows*g.Cols)
	for i := range g.Rows {
		for j := range g.Cols {
			g.Data = append(g.Data, complex128(t.At(i, j)))
		}
	}
	return g
}

func gemm(ts ...*Dense) *Dense {
	t128 := to128(ts[0])
	for i := 1; i < len(ts); i++ {
		t1 := to128(Zeros(t128.Rows, ts[i].Shape()[1]))
		cblas128.Gemm(blas.NoTrans, blas.NoTrans, 1, t128, to128(ts[i]), 0, t1)
		t128 = t1
	}

	t := Zeros(t128.Rows, t128.Cols)
	for i := range t128.Rows {
		for j := range t128.Cols {
			t.SetAt([]int{i, j}, complex64(t128.Data[i*t128.Cols+j]))
		}
	}
	return t
}
