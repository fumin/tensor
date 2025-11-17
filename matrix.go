package tensor

import (
	"cmp"
	"fmt"
	"math"
	"math/cmplx"
)

// Eye creates an identity matrix with ones at the k-th diagonal.
func (t *Dense) Eye(n, k int) *Dense {
	t.Reset(n, n)
	for i := range n {
		j := i + k
		if !(j >= 0 && j < n) {
			continue
		}

		ptr := i*t.shape[1] + j
		t.data[ptr] = 1
	}
	return t
}

// InfNorm returns the infinity norm of matrix a.
func (a *Dense) InfNorm() float32 {
	var norm float32 = -1
	for i := 0; i < a.Shape()[0]; i++ {
		var ni float32
		for j := 0; j < a.Shape()[1]; j++ {
			ni += abs(a.At(i, j))
		}
		norm = max(ni, norm)
	}
	return norm
}

func eig22(t *Dense) (complex64, complex64) {
	a, b := complex128(t.At(0, 0)), complex128(t.At(0, 1))
	c, d := complex128(t.At(1, 0)), complex128(t.At(1, 1))
	iSqrt := cmplx.Sqrt(a*a - 2*a*d + 4*b*c + d*d)
	return complex64(0.5 * (-iSqrt + a + d)), complex64(0.5 * (iSqrt + a + d))
}

// svd22 computes the singular value decomposition for 2x2 matrices.
// See Closed Form SVD Solutions for 2 x 2 Matrices, John J. Polcari.
func svd22(s, u, v *Dense) {
	sign := func(x float64) complex128 { return complex(float64(cmp.Compare(x, 0)), 0) }

	a, phiA := cmplx.Polar(complex128(s.At(0, 0)))
	b, phiB := cmplx.Polar(complex128(s.At(1, 0)))
	c, phiC := cmplx.Polar(complex128(s.At(0, 1)))
	d, phiD := cmplx.Polar(complex128(s.At(1, 1)))

	// Handle the case where a == 0 && d == 0, resulting in singularity in Atan2(, cosThetaR), due to cosThetaR == 0.
	eps := float64(epsilon)
	if math.Abs(a)+math.Abs(d) < eps*(math.Abs(b)+math.Abs(c)) {
		if math.Abs(c) > math.Abs(b) {
			u.SetAt([]int{0, 0}, 1)
			u.SetAt([]int{0, 1}, 0)
			u.SetAt([]int{1, 0}, 0)
			u.SetAt([]int{1, 1}, 1)

			s.SetAt([]int{0, 0}, complex64(complex(math.Abs(c), 0)))
			s.SetAt([]int{0, 1}, 0)
			s.SetAt([]int{1, 0}, 0)
			s.SetAt([]int{1, 1}, complex64(complex(math.Abs(b), 0)))

			v.SetAt([]int{0, 0}, 0)
			v.SetAt([]int{0, 1}, complex64(cmplx.Rect(1, -phiB)))
			v.SetAt([]int{1, 0}, complex64(cmplx.Rect(1, -phiC)))
			v.SetAt([]int{1, 1}, 0)
		} else {
			u.SetAt([]int{0, 0}, 0)
			u.SetAt([]int{0, 1}, 1)
			u.SetAt([]int{1, 0}, 1)
			u.SetAt([]int{1, 1}, 0)

			s.SetAt([]int{0, 0}, complex64(complex(math.Abs(b), 0)))
			s.SetAt([]int{0, 1}, 0)
			s.SetAt([]int{1, 0}, 0)
			s.SetAt([]int{1, 1}, complex64(complex(math.Abs(c), 0)))

			v.SetAt([]int{0, 0}, complex64(cmplx.Rect(1, -phiB)))
			v.SetAt([]int{0, 1}, 0)
			v.SetAt([]int{1, 0}, 0)
			v.SetAt([]int{1, 1}, complex64(cmplx.Rect(1, -phiC)))
		}
		return
	}

	// Handle the case b == 0 && c == 0, resulting in singularity in Atan2(, cosThetaR), due to cosThetaR == 0.
	// In this case, we only need to slightly perturb the off-diagonals.
	if math.Abs(b)+math.Abs(c) < eps*(math.Abs(a)+math.Abs(d)) {
		perturbation := float64(epsilon) / 128
		if math.Abs(b) < perturbation {
			b = perturbation
		}
	}

	// Compute singular values, lambda1 and lambda2.
	sum2 := a*a + b*b + c*c + d*d
	deltaPhi := phiA - phiB - phiC + phiD
	abcdCos := a * b * c * d * (1 - math.Cos(deltaPhi))
	z := math.Sqrt(math.Pow(a*d-b*c, 2) + 2*abcdCos)
	lambda1 := (math.Sqrt(sum2+2*z) + math.Sqrt(sum2-2*z)) / 2
	lambda2 := math.Abs(math.Sqrt(sum2+2*z)-math.Sqrt(sum2-2*z)) / 2

	// Compute the singular value basis.
	cosThetaL := 2 * math.Sqrt(math.Pow(a*b+c*d, 2)-2*abcdCos)
	thetaL := math.Atan2(-(a*a-d*d)+(b*b-c*c)+math.Sqrt(sum2*sum2-4*z*z), cosThetaL)
	cosThetaR := 2 * math.Sqrt(math.Pow(a*c+b*d, 2)-2*abcdCos)
	thetaR := math.Atan2(-(a*a-d*d)-(b*b-c*c)+math.Sqrt(sum2*sum2-4*z*z), cosThetaR)
	phiL := cmplx.Phase(complex(a*b, 0)*cmplx.Rect(1, phiB-phiA) + complex(c*d, 0)*cmplx.Rect(1, phiD-phiC))
	phiR := cmplx.Phase(complex(a*c, 0)*cmplx.Rect(1, phiA-phiC) + complex(b*d, 0)*cmplx.Rect(1, phiB-phiD))
	omega1 := cmplx.Phase(sign(a-d) * (complex(a*math.Cos(thetaL)*math.Cos(thetaR), 0)*cmplx.Rect(1, phiA) - complex(d*math.Sin(thetaL)*math.Sin(thetaR), 0)*cmplx.Rect(1, phiD-phiL+phiR)))
	omega2 := cmplx.Phase(sign(a-d) * (complex(d*math.Cos(thetaL)*math.Cos(thetaR), 0)*cmplx.Rect(1, phiD) - complex(a*math.Sin(thetaL)*math.Sin(thetaR), 0)*cmplx.Rect(1, phiA+phiL-phiR)))

	s.SetAt([]int{0, 0}, complex64(complex(lambda1, 0)))
	s.SetAt([]int{0, 1}, 0)
	s.SetAt([]int{1, 0}, 0)
	s.SetAt([]int{1, 1}, complex64(complex(lambda2, 0)))

	u.SetAt([]int{0, 0}, complex64(complex(math.Cos(thetaL), 0)))
	u.SetAt([]int{0, 1}, -1*complex64(complex(math.Sin(thetaL), 0)*cmplx.Rect(1, -phiL)))
	u.SetAt([]int{1, 0}, complex64(complex(math.Sin(thetaL), 0)*cmplx.Rect(1, phiL)))
	u.SetAt([]int{1, 1}, complex64(complex(math.Cos(thetaL), 0)))

	v.SetAt([]int{0, 0}, complex64(complex(math.Cos(thetaR), 0)*cmplx.Rect(1, omega1)))
	v.SetAt([]int{0, 1}, complex64(complex(math.Sin(thetaR), 0)*cmplx.Rect(1, -phiR)*cmplx.Rect(1, omega1)))
	v.SetAt([]int{1, 0}, -1*complex64(complex(math.Sin(thetaR), 0)*cmplx.Rect(1, phiR)*cmplx.Rect(1, omega2)))
	v.SetAt([]int{1, 1}, complex64(complex(math.Cos(thetaR), 0)*cmplx.Rect(1, omega2)))

	// Now A = U @ S @ V, make v adjoint to follow convention U @ S @ V.H.
	v.SetAt([]int{0, 0}, conj(v.At(0, 0)))
	v01 := v.At(0, 1)
	v.SetAt([]int{0, 1}, conj(v.At(1, 0)))
	v.SetAt([]int{1, 0}, conj(v01))
	v.SetAt([]int{1, 1}, conj(v.At(1, 1)))
}

// Triu zeros elements below the k-th diagonal.
func (t *Dense) Triu(k int) *Dense {
	d := t.digits[:t.dimension]
	t.initDigits()
	for t.incrDigits() {
		i, j := d[len(d)-2], d[len(d)-1]
		if j < i+k {
			t.SetAt(d, 0)
		}
	}
	return t
}

// Tril zeros elements above the k-th diagonal.
func (t *Dense) Tril(k int) *Dense {
	d := t.digits[:t.dimension]
	t.initDigits()
	for t.incrDigits() {
		i, j := d[len(d)-2], d[len(d)-1]
		if j > i+k {
			t.SetAt(d, 0)
		}
	}
	return t
}

// Exp computes the matrix exponential of a.
func Exp(a *Dense, buf [4]*Dense) *Dense {
	j := max(0, 1+int(math.Floor(math.Log2(float64(a.InfNorm())))))
	j2 := 1 << j
	a.Mul(complex(1./float32(j2), 0))
	const q = 6
	d := buf[0].Eye(a.Shape()[0], 0)
	n := buf[1].Eye(a.Shape()[0], 0)
	x := buf[2].Eye(a.Shape()[0], 0)
	var c complex64 = 1
	for ki := 1; ki <= q; ki++ {
		k := complex(float32(ki), 0)
		c = c * (q - k + 1) / ((2*q - k + 1) * k)
		x.Set(nil, MatMul(buf[3], a, x))
		n.Add(c, x)
		if ki%2 == 1 {
			d.Add(-c, x)
		} else {
			d.Add(c, x)
		}
	}

	// Solve for f in d @ f = n.
	u := buf[2]
	qrBufs := [2]*Dense{a, buf[3]}
	d = QR(u, d, qrBufs)
	n = MatMul(buf[3], u.H(), n)
	f := a
	for j := range f.Shape()[1] {
		bnd := [][2]int{{}, {j, j + 1}}
		fj := f.Slice(bnd)
		nj := n.Slice(bnd)
		backSubstitution(fj, d, nj, -1)
	}

	for range j {
		f.Set(nil, MatMul(buf[0], f, f))
	}
	return f
}

func matmul(c, a, b *Dense) *Dense {
	m, an, n := a.Shape()[0], a.Shape()[1], b.Shape()[1]
	if an != b.Shape()[0] {
		panic(fmt.Sprintf("matmul wrong shapes %#v %#v", a.Shape(), b.Shape()))
	}
	c.Reset(m, n)
	adata, bdata := a.data[a.axis[1].start:], b.data[b.axis[1].start:]
	aStride, bStride := a.axis[1].size, b.axis[1].size
	if a.axisToView[0] == 0 {
		if b.axisToView[0] == 0 {
			if a.conj {
				if b.conj {
					for i := range m {
						for j := range n {
							ap := i * aStride
							bp := j
							var v complex64
							for _ = range an {
								av := conj(adata[ap])
								bv := conj(bdata[bp])
								v += av * bv

								ap++
								bp += bStride
							}
							c.data[i*n+j] = v
						}
					}
				} else {
					for i := range m {
						for j := range n {
							ap := i * aStride
							bp := j
							var v complex64
							for _ = range an {
								av := conj(adata[ap])
								bv := bdata[bp]
								v += av * bv

								ap++
								bp += bStride
							}
							c.data[i*n+j] = v
						}
					}
				}
			} else {
				if b.conj {
					for i := range m {
						for j := range n {
							ap := i * aStride
							bp := j
							var v complex64
							for _ = range an {
								av := adata[ap]
								bv := conj(bdata[bp])
								v += av * bv

								ap++
								bp += bStride
							}
							c.data[i*n+j] = v
						}
					}
				} else {
					for i := range m {
						for j := range n {
							ap := i * aStride
							bp := j
							var v complex64
							for _ = range an {
								av := adata[ap]
								bv := bdata[bp]
								v += av * bv

								ap++
								bp += bStride
							}
							c.data[i*n+j] = v
						}
					}
				}
			}
		} else {
			if a.conj {
				if b.conj {
					for i := range m {
						for j := range n {
							ap := i * aStride
							bp := j * bStride
							var v complex64
							for _ = range an {
								av := conj(adata[ap])
								bv := conj(bdata[bp])
								v += av * bv

								ap++
								bp++
							}
							c.data[i*n+j] = v
						}
					}

				} else {
					for i := range m {
						for j := range n {
							ap := i * aStride
							bp := j * bStride
							var v complex64
							for _ = range an {
								av := conj(adata[ap])
								bv := bdata[bp]
								v += av * bv

								ap++
								bp++
							}
							c.data[i*n+j] = v
						}
					}
				}
			} else {
				if b.conj {
					for i := range m {
						for j := range n {
							ap := i * aStride
							bp := j * bStride
							var v complex64
							for _ = range an {
								av := adata[ap]
								bv := conj(bdata[bp])
								v += av * bv

								ap++
								bp++
							}
							c.data[i*n+j] = v
						}
					}
				} else {
					for i := range m {
						for j := range n {
							ap := i * aStride
							bp := j * bStride
							var v complex64
							for _ = range an {
								av := adata[ap]
								bv := bdata[bp]
								v += av * bv

								ap++
								bp++
							}
							c.data[i*n+j] = v
						}
					}
				}
			}
		}
	} else {
		if b.axisToView[0] == 0 {
			if a.conj {
				if b.conj {
					for i := range m {
						for j := range n {
							ap := i
							bp := j
							var v complex64
							for _ = range an {
								av := conj(adata[ap])
								bv := conj(bdata[bp])
								v += av * bv

								ap += aStride
								bp += bStride
							}
							c.data[i*n+j] = v
						}
					}
				} else {
					for i := range m {
						for j := range n {
							ap := i
							bp := j
							var v complex64
							for _ = range an {
								av := conj(adata[ap])
								bv := bdata[bp]
								v += av * bv

								ap += aStride
								bp += bStride
							}
							c.data[i*n+j] = v
						}
					}
				}
			} else {
				if b.conj {
					for i := range m {
						for j := range n {
							ap := i
							bp := j
							var v complex64
							for _ = range an {
								av := adata[ap]
								bv := conj(bdata[bp])
								v += av * bv

								ap += aStride
								bp += bStride
							}
							c.data[i*n+j] = v
						}
					}
				} else {
					for i := range m {
						for j := range n {
							ap := i
							bp := j
							var v complex64
							for _ = range an {
								av := adata[ap]
								bv := bdata[bp]
								v += av * bv

								ap += aStride
								bp += bStride
							}
							c.data[i*n+j] = v
						}
					}
				}
			}
		} else {
			if a.conj {
				if b.conj {
					for i := range m {
						for j := range n {
							ap := i
							bp := j * bStride
							var v complex64
							for _ = range an {
								av := conj(adata[ap])
								bv := conj(bdata[bp])
								v += av * bv

								ap += aStride
								bp++
							}
							c.data[i*n+j] = v
						}
					}
				} else {
					for i := range m {
						for j := range n {
							ap := i
							bp := j * bStride
							var v complex64
							for _ = range an {
								av := conj(adata[ap])
								bv := bdata[bp]
								v += av * bv

								ap += aStride
								bp++
							}
							c.data[i*n+j] = v
						}
					}
				}
			} else {
				if b.conj {
					for i := range m {
						for j := range n {
							ap := i
							bp := j * bStride
							var v complex64
							for _ = range an {
								av := adata[ap]
								bv := conj(bdata[bp])
								v += av * bv

								ap += aStride
								bp++
							}
							c.data[i*n+j] = v
						}
					}
				} else {
					for i := range m {
						for j := range n {
							ap := i
							bp := j * bStride
							var v complex64
							for _ = range an {
								av := adata[ap]
								bv := bdata[bp]
								v += av * bv

								ap += aStride
								bp++
							}
							c.data[i*n+j] = v
						}
					}
				}
			}
		}
	}

	// for i := range m {
	// 	for j := range n {
	// 		var ap int
	// 		if a.axisToView[0] == 0 {
	// 			ap = i * a.axis[1].size
	// 			ap--
	// 		} else { // Transposed.
	// 			ap = i
	// 			ap -= a.axis[1].size
	// 		}
	// 		var bp int
	// 		if b.axisToView[0] == 0 {
	// 			bp = j
	// 			bp -= b.axis[1].size
	// 		} else { // Transposed.
	// 			bp = j * b.axis[1].size
	// 			bp--
	// 		}

	// 		var v complex64
	// 		for _ = range an {
	// 			if a.axisToView[0] == 0 {
	// 				ap++
	// 			} else { // Transposed.
	// 				ap += a.axis[1].size
	// 			}
	// 			av := adata[ap]
	// 			if a.conj {
	// 				av = conj(av)
	// 			}

	// 			if b.axisToView[0] == 0 {
	// 				bp += b.axis[1].size
	// 			} else { // Transposed.
	// 				bp++
	// 			}
	// 			bv := bdata[bp]
	// 			if b.conj {
	// 				bv = conj(bv)
	// 			}

	// 			v += av * bv
	// 		}
	// 		c.data[i*n+j] = v
	// 	}
	// }
	return c
}
