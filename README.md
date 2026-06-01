# SpacetimeMetrics.jl

Analytically known metrics for general relativity.

[![CI](https://github.com/eschnett/SpacetimeMetrics.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/eschnett/SpacetimeMetrics.jl/actions/workflows/CI.yml)
[![Documentation](https://github.com/eschnett/SpacetimeMetrics.jl/actions/workflows/docs.yml/badge.svg)](https://eschnett.github.io/SpacetimeMetrics.jl/)

## Overview

Metrics:
- Minkowski
- Kerr-Schild (without charge)
- Harmonic coordinates
- Gauge wave (Apples-with-Apples testbed)

Transformations:
- Translations
- Rotations
- Boosts
- Gauge wave (time-dependent coordinate transformation)

Geometry:
- first and second derivatives of the metric
- Christoffel symbols
- Riemmann tensor, Ricci tensor, Einstein tensor

## Example

```julia
julia> using SpacetimeMetrics

julia> M=1.0; a=0.8;

julia> ks = KerrSchild(M, a);

julia> t,x,y,z = (0.0, 1.0, 1.0, 1.0);

julia> g = metric(ks, [t,x,y,z])
4×4 StaticArraysCore.SMatrix{4, 4, Float64, 16} with indices SOneTo(4)×SOneTo(4):
 0.132273  0.84222   0.284041  0.701448
 0.84222   1.62647   0.211278  0.521759
 0.284041  0.211278  1.07125   0.175965
 0.701448  0.521759  0.175965  1.43455

julia> G = EinsteinTensor(ks, [t,x,y,z])

4×4 StaticArraysCore.SMatrix{4, 4, Float64, 16} with indices SOneTo(4)×SOneTo(4):
 1.67028e-16  2.61733e-16   3.04329e-16   3.86998e-16
 2.61733e-16  2.65653e-16   3.1863e-16    5.43956e-16
 3.04329e-16  3.1863e-16   -1.3614e-16    1.86894e-16
 3.86998e-16  5.43956e-16   1.86894e-16  -3.67186e-16
```

Calling `metric` to evaluate it at a point is efficient and suitable for loop kernels.
