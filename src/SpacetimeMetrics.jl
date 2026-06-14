module SpacetimeMetrics

using ForwardDiff
using LinearAlgebra
using PrecompileTools
using StaticArrays

export AbstractMetric
"""
    AbstractMetric

Supertype for analytic 4-dimensional Lorentzian spacetime metrics.

# Conventions

These are the standard numerical-relativity conventions (Baumgarte & Shapiro,
*Numerical Relativity*; MTW/Wald for the 4-curvature). None are nonstandard for NR.

- Four spacetime dimensions; metric signature `(−1, +1, +1, +1)`; units `c = G = 1`.
- Christoffel symbols (Levi-Civita):
  `Γ^a_{bc} = ½ g^{ad}(∂_b g_{dc} + ∂_c g_{db} − ∂_d g_{bc})` — see [`ChristoffelSymbols`](@ref).
- Riemann tensor in the **MTW/Wald** sign, `[∇_c, ∇_d] V^a = R^a_{bcd} V^b`:
  `R^a_{bcd} = ∂_c Γ^a_{db} − ∂_d Γ^a_{cb} + Γ^a_{ce} Γ^e_{db} − Γ^a_{de} Γ^e_{cb}`
  — see [`RiemannTensor`](@ref). (Opposite sign to the Landau–Lifshitz/Weinberg family.)
- Ricci tensor by the first–third trace `R_{ab} = R^c_{acb}` — see [`RicciTensor`](@ref);
  with the above this gives the Ricci scalar `R > 0` on a sphere and `G_{ab} = 8π T_{ab}`.
- Extrinsic curvature `K_{ij} = −½ £_n γ_{ij}` (equivalently `K = −∇_a n^a`) — see
  [`ExtrinsicCurvature`](@ref). This is the **NR/ADM** sign; some GR texts (e.g. Wald)
  use the opposite sign `+½ £_n γ_{ij}`.
- Spatial 3-curvature [`SpatialRicciTensor`](@ref)/[`SpatialRicciScalar`](@ref) uses the
  same conventions on the slice, so `⁽³⁾R + K² − K_{ij}K^{ij} = 16π ρ` (Hamiltonian constraint).

The one quantity with no single NR standard is the generalized-harmonic gauge source,
defined here as `H^a = −Γ^a` (see [`gauge_source`](@ref)).

Concrete subtypes must implement [`metric`](@ref) and `Base.nameof`.
"""
abstract type AbstractMetric end

export metric
"""
    metric(m::AbstractMetric, x::AbstractVector) -> SMatrix{4,4}

Return the covariant metric tensor `g_{ab}(x)` at the 4-position
`x = (t, x, y, z)`. Must be implemented for every concrete subtype of
[`AbstractMetric`](@ref).
"""
metric(::AbstractMetric, x::AbstractVector) = error("not implemented")

Base.ndims(::AbstractMetric) = 4
Base.nameof(::AbstractMetric) = error("not implemented")

################################################################################

# Minkowski metric
const η = SMatrix{4,4}(-1, 0, 0, 0, 0, +1, 0, 0, 0, 0, +1, 0, 0, 0, 0, +1)

################################################################################

# Derivatives

# Tags for non-allocating AD passes — one per differentiation level
struct _DMetricTag end
struct _DDMetricTag end
struct _DChristoffelTag end
struct _DGaugeSourceTag end
struct _DSpatialChristoffelTag end

# Seed an SVector{4} with 4-component dual numbers (one partial per coordinate).
# When T is itself a Dual (nested call), one(T)/zero(T) produce the right typed values.
@inline function make_dual(p::SVector{4,T}, ::Type{Tag}) where {T,Tag}
    z, o = zero(T), one(T)
    SVector(
        ForwardDiff.Dual{Tag}(p[1], o, z, z, z),
        ForwardDiff.Dual{Tag}(p[2], z, o, z, z),
        ForwardDiff.Dual{Tag}(p[3], z, z, o, z),
        ForwardDiff.Dual{Tag}(p[4], z, z, z, o),
    )
end

################################################################################

# Geometry

export dmetric
"""
    dmetric(m::AbstractMetric, p::AbstractVector) -> (g, dg)

Return the metric `g_{ab}` and its first coordinate derivative
`dg[a, b, c] = ∂_c g_{ab}` at the 4-position `p`. Computed by forward-mode
automatic differentiation through [`metric`](@ref) with a single dual-number
seed; allocation-free and GPU-friendly.
"""
function dmetric(m::AbstractMetric, p::AbstractVector)
    p = SVector{4}(p)
    g_dual = metric(m, make_dual(p, _DMetricTag))
    g = SMatrix{4,4}(ForwardDiff.value(g_dual[a, b]) for a in 1:4, b in 1:4)
    dg = SArray{Tuple{4,4,4}}(ForwardDiff.partials(g_dual[a, b], c) for a in 1:4, b in 1:4, c in 1:4)
    return g, dg
end

export ddmetric
"""
    ddmetric(m::AbstractMetric, p::AbstractVector) -> (g, dg, ddg)

Return the metric `g_{ab}`, its first derivative `dg[a, b, c] = ∂_c g_{ab}`,
and its second derivative `ddg[a, b, c, d] = ∂_d ∂_c g_{ab}` at the 4-position
`p`. The second-derivative pass nests an additional dual layer on top of
[`dmetric`](@ref). The result is symmetrized over the trailing index pair.
"""
function ddmetric(m::AbstractMetric, p::AbstractVector)
    p = SVector{4}(p)
    # dmetric returns (g_dual, dg_dual) with Dual{_DDMetricTag,T,4} elements;
    # inside dmetric a second Dual{_DMetricTag,...} layer is added — dual-of-dual.
    g_dual, dg_dual = dmetric(m, make_dual(p, _DDMetricTag))
    g = SMatrix{4,4}(ForwardDiff.value(g_dual[a, b]) for a in 1:4, b in 1:4)
    dg = SArray{Tuple{4,4,4}}(ForwardDiff.partials(g_dual[a, b], c) for a in 1:4, b in 1:4, c in 1:4)
    ddg = SArray{Tuple{4,4,4,4}}(ForwardDiff.partials(dg_dual[a, b, c], d) for a in 1:4, b in 1:4, c in 1:4, d in 1:4)

    # Symmetrize to cancel round-off errors
    ddg = SArray{Tuple{4,4,4,4}}((ddg[a, b, c, d] + ddg[a, b, d, c]) / 2 for a in 1:4, b in 1:4, c in 1:4, d in 1:4)

    return g, dg, ddg
end

export adm_decompose
"""
    adm_decompose(g::SMatrix{4,4}) -> (α, β, γ)
    adm_decompose(m::AbstractMetric, p::AbstractVector) -> (α, β, γ)

Decompose a 4-metric into its ADM 3+1 variables: the lapse `α`
(scalar), the contravariant shift `β :: SVector{3}` (`β^i`), and the
spatial 3-metric `γ :: SMatrix{3,3}` (`γ_{ij}`), such that

`ds² = −α² dt² + γ_{ij} (dx^i + β^i dt)(dx^j + β^j dt)`,

i.e. `γ_{ij} = g_{ij}`, `β_i = g_{ti}`, `β^i = γ^{ij} β_j`, and
`α = √(−g_{tt} + β^i β_i)`. Allocation-free and AD/GPU-friendly.
"""
@inline function adm_decompose(g::SMatrix{4,4})
    γ = SMatrix{3,3}(g[i + 1, j + 1] for i in 1:3, j in 1:3)
    γu = inv(γ)
    βl = SVector{3}(g[2, 1], g[3, 1], g[4, 1])
    β = γu * βl
    α = sqrt(-g[1, 1] + dot(β, βl))
    return α, β, γ
end

adm_decompose(m::AbstractMetric, p::AbstractVector) =
    adm_decompose(metric(m, SVector{4}(p)))

export ExtrinsicCurvature
"""
    ExtrinsicCurvature(m::AbstractMetric, p::AbstractVector) -> K

Return the extrinsic curvature `K_{ij}` of the constant-`t` slice through the
4-position `p`, in the 3+1 ADM decomposition of [`metric`](@ref):
`K_{ij} = -(1/(2α)) (∂_t γ_{ij} − D_i β_j − D_j β_i)`,
where `γ_{ij} = g_{ij}`, `β_i = g_{ti}`, `α = √(−g_{tt} + β^i β_i)`, and `D`
is the Levi-Civita connection of `γ`. Symmetric in `(i, j)`.
"""
function ExtrinsicCurvature(m::AbstractMetric, p::AbstractVector)
    g, dg = dmetric(m, p)

    α, β, γ = adm_decompose(g)
    γu = inv(γ)
    βl = SVector{3}(g[2, 1], g[3, 1], g[4, 1])

    dtγ = dg[2:4, 2:4, 1]
    dγ = dg[2:4, 2:4, 2:4]
    # dβl[i, j] = ∂_j βl_i = ∂_j g_{i+1, 1}
    dβl = SMatrix{3,3}(dg[i + 1, 1, j + 1] for i in 1:3, j in 1:3)

    # Γ_abc (spatial)
    Γl = SArray{Tuple{3,3,3}}((dγ[a, b, c] + dγ[a, c, b] - dγ[b, c, a]) / 2 for a in 1:3, b in 1:3, c in 1:3)

    # Γ^a_bc (spatial)
    Γ = SArray{Tuple{3,3,3}}(sum(γu[a, x] * Γl[x, b, c] for x in 1:3) for a in 1:3, b in 1:3, c in 1:3)

    # D_i β_j + D_j β_i = ∂_i β_j + ∂_j β_i − 2 Γ^k_{ij} β_k (covariant shift)
    K = SMatrix{3,3}(
        (dtγ[i, j] - dβl[i, j] - dβl[j, i] + 2 * sum(Γ[x, i, j] * βl[x] for x in 1:3)) / (-2α) for i in 1:3, j in 1:3
    )

    # Symmetrize to cancel round-off errors
    K = (K + K') / 2

    return K::SMatrix{3,3}
end

export ChristoffelSymbols
"""
    ChristoffelSymbols(m::AbstractMetric, p::AbstractVector) -> Γ

Return the Christoffel symbols of the second kind
`Γ[a, b, c] = Γ^a_{bc} = ½ g^{ad} (∂_b g_{dc} + ∂_c g_{db} − ∂_d g_{bc})`
at the 4-position `p`. Symmetric in the lower pair `(b, c)`.
"""
function ChristoffelSymbols(m::AbstractMetric, p::AbstractVector)
    g, dg = dmetric(m, p)
    gu = inv(g)

    # Γ_abc
    Γl = SArray{Tuple{4,4,4}}((dg[a, b, c] + dg[a, c, b] - dg[b, c, a]) / 2 for a in 1:4, b in 1:4, c in 1:4)

    # Γ^a_bc (explicit four-term contraction: a generator-based `sum`
    # inside the comprehension heap-allocates and is the hot path under
    # nested dual numbers in gauge_source_grad)
    Γ = SArray{Tuple{4,4,4}}(
        gu[a, 1] * Γl[1, b, c] + gu[a, 2] * Γl[2, b, c] +
        gu[a, 3] * Γl[3, b, c] + gu[a, 4] * Γl[4, b, c]
        for a in 1:4, b in 1:4, c in 1:4)

    # Symmetrize to cancel round-off errors. NOTE the fresh name: a
    # generator that captures a variable which is REASSIGNED forces the
    # captured variable into a heap Box (classic closure-capture
    # pitfall) — `Γ = SArray((Γ[…] + Γ[…])/2 …)` allocated ~6 kB/call.
    Γs = SArray{Tuple{4,4,4}}((Γ[a, b, c] + Γ[a, c, b]) / 2
                              for a in 1:4, b in 1:4, c in 1:4)
    return Γs
end

export dChristoffelSymbols
"""
    dChristoffelSymbols(m::AbstractMetric, p::AbstractVector) -> (Γ, dΓ)

Return the Christoffel symbols `Γ[a, b, c] = Γ^a_{bc}` and their first
coordinate derivative `dΓ[a, b, c, d] = ∂_d Γ^a_{bc}` at the 4-position `p`.
"""
function dChristoffelSymbols(m::AbstractMetric, p::AbstractVector)
    p = SVector{4}(p)
    # ChristoffelSymbols calls dmetric internally, so this naturally composes to
    # dual-of-dual: _DChristoffelTag outer, _DMetricTag inner.
    Γ_dual = ChristoffelSymbols(m, make_dual(p, _DChristoffelTag))
    Γ = SArray{Tuple{4,4,4}}(ForwardDiff.value(Γ_dual[a, b, c]) for a in 1:4, b in 1:4, c in 1:4)
    dΓ = SArray{Tuple{4,4,4,4}}(ForwardDiff.partials(Γ_dual[a, b, c], d) for a in 1:4, b in 1:4, c in 1:4, d in 1:4)

    # Symmetrize to cancel round-off errors (fresh name — see the
    # closure-capture note in ChristoffelSymbols).
    dΓs = SArray{Tuple{4,4,4,4}}((dΓ[a, b, c, d] + dΓ[a, c, b, d]) / 2
                                 for a in 1:4, b in 1:4, c in 1:4, d in 1:4)
    return Γ, dΓs
end

export RiemannTensor
"""
    RiemannTensor(m::AbstractMetric, p::AbstractVector) -> R

Return the Riemann curvature tensor (mixed form) at the 4-position `p`:
`R[a, b, c, d] = R^a_{bcd} = ∂_c Γ^a_{db} − ∂_d Γ^a_{cb}
                            + Γ^a_{ce} Γ^e_{db} − Γ^a_{de} Γ^e_{cb}`.
Antisymmetric in the last pair `(c, d)`.
"""
function RiemannTensor(m::AbstractMetric, p::AbstractVector)
    Γ, dΓ = dChristoffelSymbols(m, p)

    # R^a_bcd in the MTW/Wald sign convention, [∇_c, ∇_d] V^a = R^a_{bcd} V^b
    # (verified: standard NR convention — see the `AbstractMetric` docstring).
    Rm = SArray{Tuple{4,4,4,4}}(
        dΓ[a, d, b, c] - dΓ[a, c, b, d] + sum(Γ[a, c, x] * Γ[x, d, b] - Γ[a, d, x] * Γ[x, c, b] for x in 1:4) for
        a in 1:4, b in 1:4, c in 1:4, d in 1:4
    )

    # (Anti-)symmetrize to cancel round-off errors (fresh name — see
    # the closure-capture note in ChristoffelSymbols).
    # We should probably apply more symmetries/antisymmetries
    Rms = SArray{Tuple{4,4,4,4}}((Rm[a, b, c, d] - Rm[a, b, d, c]) / 2
                                 for a in 1:4, b in 1:4, c in 1:4, d in 1:4)
    return Rms
end

export RicciTensor
"""
    RicciTensor(m::AbstractMetric, p::AbstractVector) -> Rc

Return the Ricci tensor `Rc[a, b] = R_{ab} = R^c_{acb}` at the 4-position `p`,
the trace of the Riemann tensor over the first and third indices. Symmetric.
"""
function RicciTensor(m::AbstractMetric, p::AbstractVector)
    Rm = RiemannTensor(m, p)

    # R_ab = R^x_{axb}, the first–third trace (verified: standard NR convention,
    # Ricci scalar > 0 on a sphere — see the `AbstractMetric` docstring).
    Rc = SArray{Tuple{4,4}}(sum(Rm[x, a, x, b] for x in 1:4) for a in 1:4, b in 1:4)

    # Symmetrize to cancel round-off errors
    Rc = (Rc + Rc') / 2

    return Rc::SMatrix{4,4}
end

export EinsteinTensor
"""
    EinsteinTensor(m::AbstractMetric, p::AbstractVector) -> G

Return the Einstein tensor `G[a, b] = G_{ab} = R_{ab} − ½ R g_{ab}` at the
4-position `p`. Vanishes identically in vacuum.
"""
function EinsteinTensor(m::AbstractMetric, p::AbstractVector)
    g = metric(m, p)
    gu = inv(g)
    Rc = RicciTensor(m, p)

    Rs = sum(Rc[x, y] * gu[x, y] for x in 1:4, y in 1:4)

    # G_ab
    G = SArray{Tuple{4,4}}(Rc[a, b] - 1//2 * Rs * g[a, b] for a in 1:4, b in 1:4)

    # Symmetrize to cancel round-off errors
    G = (G + G') / 2

    return G::SMatrix{4,4}
end

# Christoffel symbols Γ^a_{bc} of the spatial 3-metric γ_{ij} = g_{i+1,j+1} on the
# constant-t slice (spatial indices only). Spatial analogue of the block used inside
# ExtrinsicCurvature; computed from the spatial derivatives of the 4-metric.
function _spatial_christoffel(m::AbstractMetric, p::AbstractVector)
    g, dg = dmetric(m, p)
    γu = inv(SMatrix{3,3}(g[i + 1, j + 1] for i in 1:3, j in 1:3))
    dγ = SArray{Tuple{3,3,3}}(dg[i + 1, j + 1, k + 1] for i in 1:3, j in 1:3, k in 1:3)  # ∂_k γ_ij
    Γl = SArray{Tuple{3,3,3}}((dγ[a, b, c] + dγ[a, c, b] - dγ[b, c, a]) / 2 for a in 1:3, b in 1:3, c in 1:3)
    Γ = SArray{Tuple{3,3,3}}(
        γu[a, 1] * Γl[1, b, c] + γu[a, 2] * Γl[2, b, c] + γu[a, 3] * Γl[3, b, c] for a in 1:3, b in 1:3, c in 1:3
    )
    # Symmetrize the lower pair to cancel round-off (fresh name — closure-capture pitfall).
    Γs = SArray{Tuple{3,3,3}}((Γ[a, b, c] + Γ[a, c, b]) / 2 for a in 1:3, b in 1:3, c in 1:3)
    return Γs
end

# Spatial Christoffel symbols Γ^a_{bc} and their spatial derivatives ∂_d Γ^a_{bc}.
# Like dChristoffelSymbols, but 3D and differentiating only along the slice: the dual
# seed covers all four coordinates and we keep the three spatial partials (skip ∂_t).
function _d_spatial_christoffel(m::AbstractMetric, p::AbstractVector)
    p = SVector{4}(p)
    Γ_dual = _spatial_christoffel(m, make_dual(p, _DSpatialChristoffelTag))
    Γ = SArray{Tuple{3,3,3}}(ForwardDiff.value(Γ_dual[a, b, c]) for a in 1:3, b in 1:3, c in 1:3)
    dΓ = SArray{Tuple{3,3,3,3}}(
        ForwardDiff.partials(Γ_dual[a, b, c], d + 1) for a in 1:3, b in 1:3, c in 1:3, d in 1:3
    )
    # Symmetrize the lower pair (fresh name — closure-capture pitfall).
    dΓs = SArray{Tuple{3,3,3,3}}((dΓ[a, b, c, d] + dΓ[a, c, b, d]) / 2 for a in 1:3, b in 1:3, c in 1:3, d in 1:3)
    return Γ, dΓs
end

export SpatialRicciTensor
"""
    SpatialRicciTensor(m::AbstractMetric, p::AbstractVector) -> R3

Return the Ricci tensor `R3[i, j] = ⁽³⁾R_{ij}` of the spatial 3-metric `γ_{ij}`
induced on the constant-`t` slice through the 4-position `p` — the *intrinsic*
Ricci tensor of `γ` (spatial covariant derivatives only), the spatial analogue of
[`ExtrinsicCurvature`](@ref). Symmetric in `(i, j)`.

This is the **standard NR / Baumgarte–Shapiro** spatial Ricci: the same sign and
first–third trace convention as the 4-Ricci [`RicciTensor`](@ref), restricted to the
slice — `⁽³⁾R_{ij} = ⁽³⁾R^k_{ikj}`, so `⁽³⁾R > 0` on a sphere. BS do not use a separate
3D convention; it is the literal 3D restriction of the 4-curvature formulas. With
[`ExtrinsicCurvature`](@ref) (`K = −½ £_n γ`) it gives the Hamiltonian-constraint
combination `⁽³⁾R + K² − K_{ij}K^{ij}` (`= 16π ρ`, hence `0` in vacuum).
"""
function SpatialRicciTensor(m::AbstractMetric, p::AbstractVector)
    Γ, dΓ = _d_spatial_christoffel(m, p)

    # ⁽³⁾R^a_bcd (same index convention as RiemannTensor). Explicit 3-term contraction
    # rather than a `sum` generator, which heap-allocates under nested dual numbers.
    Rm = SArray{Tuple{3,3,3,3}}(
        dΓ[a, d, b, c] - dΓ[a, c, b, d] +
        Γ[a, c, 1] * Γ[1, d, b] - Γ[a, d, 1] * Γ[1, c, b] +
        Γ[a, c, 2] * Γ[2, d, b] - Γ[a, d, 2] * Γ[2, c, b] +
        Γ[a, c, 3] * Γ[3, d, b] - Γ[a, d, 3] * Γ[3, c, b]
        for a in 1:3, b in 1:3, c in 1:3, d in 1:3
    )

    # Antisymmetrize the last pair to cancel round-off (fresh name — closure-capture pitfall).
    Rms = SArray{Tuple{3,3,3,3}}((Rm[a, b, c, d] - Rm[a, b, d, c]) / 2 for a in 1:3, b in 1:3, c in 1:3, d in 1:3)

    # R_ij = ⁽³⁾R^x_{ixj}, explicit contraction (allocation-free).
    R3 = SMatrix{3,3}(Rms[1, a, 1, b] + Rms[2, a, 2, b] + Rms[3, a, 3, b] for a in 1:3, b in 1:3)

    # Symmetrize to cancel round-off errors
    R3 = (R3 + R3') / 2

    return R3::SMatrix{3,3}
end

export SpatialRicciScalar
"""
    SpatialRicciScalar(m::AbstractMetric, p::AbstractVector) -> R

Return the spatial Ricci scalar `⁽³⁾R = γ^{ij} ⁽³⁾R_{ij}` on the constant-`t` slice
through `p` (the trace of [`SpatialRicciTensor`](@ref) with the spatial metric).
"""
function SpatialRicciScalar(m::AbstractMetric, p::AbstractVector)
    g = metric(m, SVector{4}(p))
    γu = inv(SMatrix{3,3}(g[i + 1, j + 1] for i in 1:3, j in 1:3))
    R3 = SpatialRicciTensor(m, p)
    return dot(γu, R3)          # γ^{ij} ⁽³⁾R_{ij}, allocation-free
end

################################################################################

# Transformations

struct TranslatedMetric{T,M} <: AbstractMetric
    metric::M
    distance::SVector{4,T}
end

export translate
"""
    translate(m::AbstractMetric, distance::AbstractVector) -> AbstractMetric

Return a metric whose value at `x` equals `m`'s value at `x − distance`. Useful
for off-centring a metric (e.g. moving a black hole away from the origin).
"""
function translate(metric::AbstractMetric, distance::AbstractVector)
    return TranslatedMetric(metric, SVector{4}(distance))
end

Base.nameof(tm::TranslatedMetric) = nameof(tm.metric) * ", translated by $(tm.distance)"

metric(tm::TranslatedMetric, x::AbstractVector) = metric(tm.metric, SVector{4}(x) - tm.distance)

################################################################################

struct RotatedMetric{T,M} <: AbstractMetric
    metric::M
    angles::SVector{3,T}        # ZYZ Euler angles (ψ, θ, φ)
    R::SMatrix{4,4,T,16}        # spatial rotation, identity on t
end

export rotate
"""
    rotate(m::AbstractMetric, ψ, θ, φ) -> AbstractMetric

Return a metric in which the spatial axes of `m` have been rotated by the
proper Euler angles `(ψ, θ, φ)` in ZYZ convention,
`R = R_z(ψ) · R_y(θ) · R_z(φ)`, with `R` acting as identity on `t`.

If `g` is the original metric and `R₄` is the 4×4 block matrix `diag(1, R)`,
the rotated metric at `x` is `R₄ · g(R₄ᵀ x) · R₄ᵀ`. Useful e.g. for tilting a
spinning black hole's spin axis away from `ẑ`.
"""
function rotate(m::AbstractMetric, ψ, θ, φ)
    T = promote_type(typeof(ψ), typeof(θ), typeof(φ))
    ψ, θ, φ = T(ψ), T(θ), T(φ)
    cψ, sψ = cos(ψ), sin(ψ)
    cθ, sθ = cos(θ), sin(θ)
    cφ, sφ = cos(φ), sin(φ)
    o = one(cψ)
    z = zero(cψ)
    # R = R_z(ψ) R_y(θ) R_z(φ), embedded as diag(1, R₃) in 4D; column-major.
    R = SMatrix{4,4}(
        o, z, z, z,
        z,  cψ*cθ*cφ - sψ*sφ,  sψ*cθ*cφ + cψ*sφ, -sθ*cφ,
        z, -cψ*cθ*sφ - sψ*cφ, -sψ*cθ*sφ + cψ*cφ,  sθ*sφ,
        z,  cψ*sθ,              sψ*sθ,             cθ,
    )
    return RotatedMetric(m, SVector{3}(ψ, θ, φ), R)
end

function Base.nameof(rm::RotatedMetric)
    ψ, θ, φ = rm.angles
    return nameof(rm.metric) * ", rotated by ZYZ Euler angles (ψ=$ψ, θ=$θ, φ=$φ)"
end

function metric(rm::RotatedMetric, x::AbstractVector)
    x_old = rm.R' * SVector{4}(x)
    g_old = metric(rm.metric, x_old)
    g = rm.R * g_old * rm.R'
    g = (g + g') / 2
    return g::SMatrix{4,4}
end

################################################################################

struct BoostedMetric{T,M} <: AbstractMetric
    metric::M
    velocity::SVector{3,T}      # boost velocity, |v| < 1
    Λ::SMatrix{4,4,T,16}        # pure Lorentz boost, symmetric
end

export boost
"""
    boost(m::AbstractMetric, v::AbstractVector) -> AbstractMetric

Return a metric in which the inertial frame of `m` has been Lorentz-boosted by
the velocity 3-vector `v` (geometric units, `c = 1`, so `|v| < 1` is required).

With `β = |v|`, `n̂ = v/β`, and `γ = 1/√(1 − β²)`, the pure-boost matrix is

    Λ = [ γ            γ vⱼ                  ]
        [ γ vᵢ    δᵢⱼ + (γ − 1) n̂ᵢ n̂ⱼ ]   (symmetric)

If `g` is the original metric, the boosted metric at `x` is `Λ · g(Λᵀ x) · Λᵀ`.
Because `Λ` is a Lorentz transformation, boosting leaves Minkowski invariant and
preserves vacuum (the Einstein tensor stays zero).
"""
function boost(m::AbstractMetric, v::AbstractVector)
    T = float(eltype(SVector{3}(v)))
    v = SVector{3,T}(v)
    β = norm(v)
    β < 1 || throw(ArgumentError("boost speed must satisfy |v| < 1, got |v| = $β"))

    γ = 1 / sqrt(1 - β^2)
    # δᵢⱼ + (γ−1) n̂ᵢ n̂ⱼ = δᵢⱼ + f vᵢ vⱼ with f = (γ−1)/β²; f → 1/2 as β → 0.
    f = β == 0 ? zero(T) : (γ - 1) / β^2
    γv = γ * v

    # Λ[α, μ], column-major (Λ is symmetric, so layout is unambiguous).
    Λ = SMatrix{4,4,T}(
        γ, γv[1], γv[2], γv[3],
        γv[1], 1 + f*v[1]*v[1], f*v[2]*v[1], f*v[3]*v[1],
        γv[2], f*v[1]*v[2], 1 + f*v[2]*v[2], f*v[3]*v[2],
        γv[3], f*v[1]*v[3], f*v[2]*v[3], 1 + f*v[3]*v[3],
    )
    return BoostedMetric(m, v, Λ)
end

Base.nameof(bm::BoostedMetric) = nameof(bm.metric) * ", boosted by velocity $(bm.velocity)"

function metric(bm::BoostedMetric, x::AbstractVector)
    x_old = bm.Λ' * SVector{4}(x)
    g_old = metric(bm.metric, x_old)
    g = bm.Λ * g_old * bm.Λ'
    g = (g + g') / 2
    return g::SMatrix{4,4}
end

################################################################################

struct GaugeWaveMetric{T,M} <: AbstractMetric
    metric::M
    amplitude::T                # A,  with |A| < 1
    period::T                   # d > 0
end

export gaugewave
"""
    gaugewave(m::AbstractMetric, A, d) -> AbstractMetric

Apply the time-dependent *gauge-wave* coordinate transformation to `m`. Applied
to [`Minkowski`](@ref) this produces the Apples-with-Apples gauge-wave testbed,
flat spacetime expressed in a sinusoidally oscillating, time-dependent chart.

The new coordinates `(t, x, y, z)` map to the inner metric's coordinates
`(t̂, x̂, ŷ, ẑ)` by

    t̂ = t − (A d / 4π) cos(2π(x − t)/d)
    x̂ = x + (A d / 4π) cos(2π(x − t)/d)
    ŷ = y,   ẑ = z

with Jacobian `J^α_μ = ∂x̂^α/∂x^μ`, and the metric is the pullback
`g_{μν} = J^α_μ J^β_ν ĝ_{αβ}`. For `m = Minkowski()` this gives

    ds² = −H dt² + H dx² + dy² + dz²,   H = 1 − A sin(2π(x − t)/d),

a flat (vanishing-curvature) but genuinely time-dependent metric.

The wave propagates along `x̂` and has amplitude `A` (require `|A| < 1` so that
`H > 0` and the signature is preserved) and spatial period `d > 0`.

Reference: Alcubierre et al., "Toward standard testbeds for numerical
relativity", arXiv:gr-qc/0305023, eqns. (4.3)–(4.4).
"""
function gaugewave(m::AbstractMetric, A, d)
    T = promote_type(typeof(A), typeof(d))
    A, d = T(A), T(d)
    abs(A) < 1 || throw(ArgumentError("gauge-wave amplitude must satisfy |A| < 1, got A = $A"))
    d > 0 || throw(ArgumentError("gauge-wave period must satisfy d > 0, got d = $d"))
    return GaugeWaveMetric(m, A, d)
end

Base.nameof(gw::GaugeWaveMetric) = nameof(gw.metric) * ", gauge wave (A=$(gw.amplitude), d=$(gw.period))"

function metric(gw::GaugeWaveMetric, x::AbstractVector)
    x = SVector{4}(x)
    A = gw.amplitude
    d = gw.period
    t, X, y, z = x

    φ = 2 * oftype(X, π) * (X - t) / d
    C = A * d / (4 * oftype(A, π))

    # Inner (e.g. inertial) coordinates Φ(x)
    cφ = cos(φ)
    x_old = SVector(t - C * cφ, X + C * cφ, y, z)
    g_old = metric(gw.metric, x_old)

    # Jacobian J[α, μ] = ∂x̂^α/∂x^μ, column-major (column = derivative variable);
    # only the (t, x) block is non-trivial, with s = (A/2) sin φ.
    s = (A / 2) * sin(φ)
    o = one(s)
    ze = zero(s)
    J = SMatrix{4,4}(
        1 - s, s, ze, ze,
        s, 1 - s, ze, ze,
        ze, ze, o, ze,
        ze, ze, ze, o,
    )

    g = J' * g_old * J
    g = (g + g') / 2
    return g::SMatrix{4,4}
end

################################################################################

struct SineShiftMetric{T,M} <: AbstractMetric
    metric::M
    amplitude::T                # A,  with |A| < 1
    period::T                   # d > 0
end

export sineshift
"""
    sineshift(m::AbstractMetric, A, d) -> AbstractMetric

Apply a time-dependent sinusoidal *spatial* reparametrisation to `m`:
the new coordinates `(t, x, y, z)` map to the inner metric's
coordinates by

    t̂ = t,   x̂ = x + (A d / 2π) sin(2π(x − t)/d),   ŷ = y,   ẑ = z

with Jacobian `J^α_μ = ∂x̂^α/∂x^μ`; the metric is the pullback
`g_{μν} = J^α_μ J^β_ν ĝ_{αβ}`. For `m = Minkowski()` this gives,
with `c = cos(2π(x − t)/d)`,

    α = 1,   βˣ = −A c / (1 + A c),   γ_xx = (1 + A c)²

— a flat metric with genuinely space- and time-varying shift and
spatial metric but unit lapse (complementary to [`gaugewave`](@ref),
which varies the lapse but has no shift). Require `|A| < 1` so the
map stays invertible (`∂x̂/∂x = 1 + A c > 0`).
"""
function sineshift(m::AbstractMetric, A, d)
    T = promote_type(typeof(A), typeof(d))
    A, d = T(A), T(d)
    abs(A) < 1 || throw(ArgumentError("sine-shift amplitude must satisfy |A| < 1, got A = $A"))
    d > 0 || throw(ArgumentError("sine-shift period must satisfy d > 0, got d = $d"))
    return SineShiftMetric(m, A, d)
end

Base.nameof(sw::SineShiftMetric) = nameof(sw.metric) * ", sine shift (A=$(sw.amplitude), d=$(sw.period))"

function metric(sw::SineShiftMetric, x::AbstractVector)
    x = SVector{4}(x)
    A = sw.amplitude
    d = sw.period
    t, X, y, z = x

    φ = 2 * oftype(X, π) * (X - t) / d
    C = A * d / (2 * oftype(A, π))

    # Inner coordinates.
    x_old = SVector(t, X + C * sin(φ), y, z)
    g_old = metric(sw.metric, x_old)

    # Jacobian J[α, μ] = ∂x̂^α/∂x^μ, column-major (column = derivative
    # variable); only the x̂ row is non-trivial, with c = A cos φ.
    c = A * cos(φ)
    o = one(c)
    ze = zero(c)
    J = SMatrix{4,4}(
        o, -c, ze, ze,
        ze, 1 + c, ze, ze,
        ze, ze, o, ze,
        ze, ze, ze, o,
    )

    g = J' * g_old * J
    g = (g + g') / 2
    return g::SMatrix{4,4}
end

export SineShift
"""
    SineShift(A, d)

Flat Minkowski spacetime in a chart with a time-dependent sinusoidal
spatial reparametrisation: unit lapse, shift
`βˣ = −A c/(1 + A c)`, spatial metric `γ_xx = (1 + A c)²` with
`c = cos(2π(x − t)/d)`. Shorthand for
`sineshift(Minkowski(), A, d)`; see [`sineshift`](@ref).
"""
SineShift(A, d) = sineshift(Minkowski(), A, d)

################################################################################

struct ShiftedMinkowskiMetric{T,M} <: AbstractMetric
    metric::M
    amplitude::T                # A,  peak |ψ′| (|A|<1); shift is always subluminal
    width::T                    # w > 0
end

export shiftedminkowski
"""
    shiftedminkowski(m::AbstractMetric, A, w) -> AbstractMetric

Skew the *time* coordinate of `m` by `t̂ = t + ψ(x)`, with the static,
x-only profile `ψ(x) = A w tanh(x/w)` (so `ψ′(x) = A sech²(x/w)`, localised
near `x = 0`). The Jacobian `J^α_μ = ∂x̂^α/∂x^μ` has the single off-diagonal
`∂t̂/∂x = ψ′`; the metric is the pullback `g_{μν} = J^α_μ J^β_ν ĝ_{αβ}`.

For `m = Minkowski()` this is flat with

    g_tt = −1,  g_tx = −ψ′(x),  g_xx = 1 − ψ′(x)²,  g_yy = g_zz = 1,

i.e. `α = 1/√(1−ψ′²)`, `βˣ = −ψ′/(1−ψ′²)`, `γ_xx = 1−ψ′²` (spacelike for
`|ψ′|<1`, so require `|A|<1`). A flat, **static (x-only)** metric with a
space-varying shift, but the shift is always **SUBLUMINAL**: `α² − |β|²_γ = 1`
*identically* (true of any static coordinate transformation of Minkowski —
`∂_t` stays timelike), so it cannot produce an outflow/excision face. A
superluminal shift needs curvature or time-dependence (see [`movinggrid`](@ref)).
NOT harmonic (`□t = −ψ″ ≠ 0` where `ψ′` varies) ⇒ to evolve it as a stationary
generalized-harmonic solution supply the gauge source [`gauge_source`](@ref).
"""
function shiftedminkowski(m::AbstractMetric, A, w)
    T = promote_type(typeof(A), typeof(w))
    A, w = T(A), T(w)
    abs(A) < 1 || throw(ArgumentError("shifted-Minkowski amplitude must satisfy |A| < 1, got A = $A"))
    w > 0 || throw(ArgumentError("shifted-Minkowski width must satisfy w > 0, got w = $w"))
    return ShiftedMinkowskiMetric(m, A, w)
end

Base.nameof(sm::ShiftedMinkowskiMetric) =
    nameof(sm.metric) * ", shifted Minkowski (A=$(sm.amplitude), w=$(sm.width))"

function metric(sm::ShiftedMinkowskiMetric, x::AbstractVector)
    x = SVector{4}(x)
    A = sm.amplitude; w = sm.width
    t, X, y, z = x
    th = tanh(X / w)
    ψ  = A * w * th
    ψ′ = A * (1 - th^2)                       # A sech²(X/w)
    x_old = SVector(t + ψ, X, y, z)
    g_old = metric(sm.metric, x_old)
    o = one(ψ′); ze = zero(ψ′)
    # J[α, μ] = ∂x̂^α/∂x^μ, column-major (column = derivative variable);
    # t̂ = t + ψ(x) ⇒ the only off-diagonal is J[1, 2] = ∂t̂/∂x = ψ′.
    J = SMatrix{4,4}(
        o,  ze, ze, ze,
        ψ′, o,  ze, ze,
        ze, ze, o,  ze,
        ze, ze, ze, o,
    )
    g = J' * g_old * J
    g = (g + g') / 2
    return g::SMatrix{4,4}
end

export ShiftedMinkowski
"""
    ShiftedMinkowski(A, w)

Flat Minkowski in time-skewed coordinates `t̂ = t + A w tanh(x/w)` — a static,
x-only metric with a space-varying but **subluminal** shift (`α²−|β|²_γ=1`).
Shorthand for `shiftedminkowski(Minkowski(), A, w)`; see [`shiftedminkowski`](@ref).
"""
ShiftedMinkowski(A, w) = shiftedminkowski(Minkowski(), A, w)

export gauge_source
"""
    gauge_source(m::AbstractMetric, p::AbstractVector) -> SVector{4}

The generalized-harmonic gauge source `H^a = −Γ^a = −g^{bc} Γ^a_{bc}` (minus the
contracted Christoffel) at `p`. With this `H^a` the metric satisfies the GH
gauge/constraint condition `C^a = Γ^a + H^a = 0`, i.e. `m` is a stationary
solution of the generalized-harmonic system driven by `H^a`. Use it to evolve
non-harmonic exact solutions (e.g. [`shiftedminkowski`](@ref), Kerr-Schild).
"""
function gauge_source(m::AbstractMetric, p::AbstractVector)
    g, _ = dmetric(m, SVector{4}(p))
    gu = inv(g)
    Γ = ChristoffelSymbols(m, p)
    # Explicit accumulation (allocation-free under dual numbers; a
    # generator-based `sum` heap-allocates here).
    return SVector{4}(ntuple(Val(4)) do a
        s = zero(eltype(g))
        @inbounds for b in 1:4, c in 1:4
            s += gu[b, c] * Γ[a, b, c]
        end
        -s
    end)
end

export gauge_source_grad
"""
    gauge_source_grad(m::AbstractMetric, p::AbstractVector) -> (Hl, dHl)

Return the gauge-source one-form lowered, `Hl_b = g_{bc} H^c` with
`H^c = −Γ^c` (see [`gauge_source`](@ref)), and its gradient
`dHl[a, b] = ∂_a Hl_b`, both at `p`. The gradient is obtained by forward-mode
differentiation through `g·gauge_source`; the inner `dmetric`/`ChristoffelSymbols`
already use a distinct dual tag, so this nests cleanly (cf. [`ddmetric`](@ref)).
These are exactly the two fields the conservative generalized-harmonic source
needs to carry a prescribed gauge source `H^a`.
"""
function gauge_source_grad(m::AbstractMetric, p::AbstractVector)
    p = SVector{4}(p)
    # Single dual-number seed through Hl_b(q) = g_{bc}(q) H^c(q), the
    # same pattern as `dmetric` (allocation-free; ForwardDiff.jacobian's
    # closure path is not).
    q = make_dual(p, _DGaugeSourceTag)
    Hd = metric(m, q) * gauge_source(m, q)
    Hl = SVector{4}(ForwardDiff.value(Hd[b]) for b in 1:4)
    dHl = SMatrix{4,4}(ForwardDiff.partials(Hd[b], a)
                       for a in 1:4, b in 1:4)         # dHl[a, b] = ∂_a Hl_b
    return Hl, dHl
end

################################################################################

struct MovingGridMetric{T,M} <: AbstractMetric
    metric::M
    speed::T                    # V₀, peak grid speed; superluminal where V(x)>1
    width::T                    # w > 0
    center::T                   # x_c (transition location)
end

export movinggrid
"""
    movinggrid(m::AbstractMetric, V₀, w, x_c) -> AbstractMetric

Pull `m` back through the **time-dependent** grid map `x̂ = x + t V(x)`,
`t̂ = t`, with the monotone profile `V(x) = ½ V₀ (1 − tanh((x−x_c)/w))`
(→ `V₀` as `x → −∞`, → 0 as `x → +∞`). The Jacobian `J^α_μ = ∂x̂^α/∂x^μ` has
`∂x̂/∂t = V` and `∂x̂/∂x = 1 + t V′`; the metric is `g = Jᵀ ĝ J`.

For `m = Minkowski()` this is flat (an exact coordinate transformation of
Minkowski) and at `t = 0` reduces to

    g_tt = V(x)² − 1,  g_tx = V(x),  g_xx = 1,  g_yy = g_zz = 1,

so `α = 1`, `βˣ = V(x)`, `γ_xx = 1`, and `α² − |β|²_γ = 1 − V²`: the shift is
**SUPERLUMINAL** wherever `V(x) > 1` (both wave characteristics `c = −V ± 1 < 0`
travel in `−x` ⇒ the `−x` face is pure outflow / excision), Minkowski where
`V → 0`. The constant-`V` case is the classic stable shifted-flat excision test;
varying `V(x)` exercises the nonlinear (varying-coefficient) source. Being
time-dependent it is *not* static — linearise at `t = 0`. NOT harmonic ⇒ supply
[`gauge_source`](@ref)/[`gauge_source_grad`](@ref) to evolve it.
"""
function movinggrid(m::AbstractMetric, V₀, w, x_c)
    T = promote_type(typeof(V₀), typeof(w), typeof(x_c))
    V₀, w, x_c = T(V₀), T(w), T(x_c)
    w > 0 || throw(ArgumentError("moving-grid width must satisfy w > 0, got w = $w"))
    return MovingGridMetric(m, V₀, w, x_c)
end

Base.nameof(mg::MovingGridMetric) =
    nameof(mg.metric) * ", moving grid (V₀=$(mg.speed), w=$(mg.width), x_c=$(mg.center))"

function metric(mg::MovingGridMetric, x::AbstractVector)
    x = SVector{4}(x)
    V₀ = mg.speed; w = mg.width; xc = mg.center
    t, X, y, z = x
    th = tanh((X - xc) / w)
    V  = V₀ * (1 - th) / 2                     # ½V₀(1−tanh((x−x_c)/w))
    Vp = -V₀ * (1 - th^2) / (2 * w)            # dV/dX
    x_old = SVector(t, X + t * V, y, z)
    g_old = metric(mg.metric, x_old)
    o = one(V); ze = zero(V)
    s = 1 + t * Vp                              # ∂x̂/∂x
    # J[α, μ] = ∂x̂^α/∂x^μ, column-major: col t = (1, V, 0, 0), col x = (0, s, 0, 0).
    J = SMatrix{4,4}(
        o,  V,  ze, ze,
        ze, s,  ze, ze,
        ze, ze, o,  ze,
        ze, ze, ze, o,
    )
    g = J' * g_old * J
    g = (g + g') / 2
    return g::SMatrix{4,4}
end

export MovingGrid
"""
    MovingGrid(V₀, w, x_c)

Flat Minkowski in a faster-than-light moving-grid chart `x̂ = x + t V(x)` with
`V(x) = ½ V₀ (1 − tanh((x−x_c)/w))` — at `t = 0`, a metric with shift
`βˣ = V(x)` that is superluminal (outflow `−x` face) where `V > 1`. Shorthand
for `movinggrid(Minkowski(), V₀, w, x_c)`; see [`movinggrid`](@ref).
"""
MovingGrid(V₀, w, x_c) = movinggrid(Minkowski(), V₀, w, x_c)

################################################################################

export Minkowski
"""
    Minkowski()

The flat Minkowski metric `η = diag(-1, +1, +1, +1)`.
"""
struct Minkowski <: AbstractMetric end

Base.nameof(::Minkowski) = "Minkowski metric"

metric(::Minkowski, ::AbstractVector{T}) where {T} = SMatrix{4,4,T}(η)

export GaugeWave
"""
    GaugeWave(A, d)

The Apples-with-Apples gauge-wave testbed: flat Minkowski spacetime in a
time-dependent, sinusoidally oscillating chart,

    ds² = −H dt² + H dx² + dy² + dz²,   H = 1 − A sin(2π(x − t)/d).

Shorthand for `gaugewave(Minkowski(), A, d)`; see [`gaugewave`](@ref) for the
underlying coordinate transformation and parameter constraints.

Reference: Alcubierre et al., "Toward standard testbeds for numerical
relativity", arXiv:gr-qc/0305023, eqns. (4.3)–(4.4).
"""
GaugeWave(A, d) = gaugewave(Minkowski(), A, d)

################################################################################

# This is not a metric -- just for testing!
# Do not export this as a metric.
struct Polynomial{T} <: AbstractMetric
    c0::T
    c1::SVector{4,T}
    c2::SMatrix{4,4,T}
end

function metric(poly::Polynomial, x::AbstractVector)
    x = SVector{4}(x)
    q = poly.c0 + sum(poly.c1[a] * x[a] for a in 1:4) + sum(1//2 * poly.c2[a, b] * x[a] * x[b] for a in 1:4, b in 1:4)
    g = SMatrix{4,4}(q, q, q, q, q, q, q, q, q, q, q, q, q, q, q, q)
    return g::SMatrix{4,4}
end

################################################################################

export KerrSchild
"""
    KerrSchild(M, a=0, Q=0)
    KerrSchild{T}(M, a=0, Q=0)

Kerr-Newman black hole in **Kerr-Schild** Cartesian coordinates `(t, x, y, z)`.

The metric is `g = η + f k ⊗ k`, where `k` is the principal null vector and
`f = (2Mr − Q²)/ρ²`, with `r` the spheroidal radius implicitly defined by
`r⁴ − r²(x²+y²+z²−a²) − a²z² = 0`.

Parameters: mass `M > 0`, spin `|a| < M`, charge `Q` (currently restricted to
`Q = 0`). Regular on the z-axis and across the future event horizon.

Reference: Cook, *Initial Data for Numerical Relativity*, §3.3.1.
"""
struct KerrSchild{T} <: AbstractMetric
    mass::T                     # 0 < M
    spin::T                     # -M < a < +M
    charge::T                   # Q
    KerrSchild{T}(mass::T, spin::T, charge::T) where {T} = new{T}(mass, spin, charge)
end
function KerrSchild{T}(mass, spin=zero(T), charge=zero(T)) where {T}
    KerrSchild{T}(T(mass), T(spin), T(charge))
end
function KerrSchild(mass, spin=0, charge=0)
    T = promote_type(typeof(mass), typeof(spin), typeof(charge))
    KerrSchild{T}(mass, spin, charge)
end

Base.nameof(ks::KerrSchild) = "Kerr-Schild metric (M=$(ks.mass), a=$(ks.spin), Q=$(ks.charge))"

function metric(ks::KerrSchild, p::AbstractVector)
    M = ks.mass
    a = ks.spin
    Q = ks.charge

    @assert length(p) == 4
    t, x, y, z = p

    # # Gegory B. Cook, "Initial Data for Numerical Relativity", Living
    # # Rev. Relativ. 3, 5 (2000), DOI:10.12942/lrr-2000-5.
    # 
    # xyz2 = x^2 + y^2 + z^2
    # 
    # r = sqrt(1//2 * (xyz2 - a^2 + sqrt(4 * a^2 * z^2 + (xyz2 - a^2)^2))) # (93)
    # θ = acos(z / r)             # (92)
    # 
    # ρ = sqrt(r^2 + a^2 * cos(θ)^2) # (79)
    # Δ = r^2 - 2M*r + a^2 + Q^2     # (79)
    # 
    # α = inv(sqrt(1 + (2M*r - Q^2) / ρ^2))         # (86)
    # βr = α^2 * (2M*r - Q^2) / ρ^2                 # (87)
    # γrr = 1 + (2M*r - Q^2) / ρ^2                  # (88)
    # γrϕ = - (1 + (2M*r - Q^2) / ρ^2) * a * sin(θ) # (89)
    # γθθ = ρ^2                                     # (90)
    # γϕϕ = γθθ * sin(θ)^2                          # (91)

    r = sqrt(1//2 * (x^2 + y^2 + z^2 - a^2 + sqrt(4 * a^2 * z^2 + (x^2 + y^2 + z^2 - a^2)^2)))

    @assert Q == 0              # TODO

    f = 2M * r^3 / (r^4 + a^2 * z^2)
    k = SVector{4}(1, (r*x + a*y) / (r^2 + a^2), (r*y - a*x) / (r^2 + a^2), z / r)
    g = SMatrix{4,4}(η[a, b] + f * k[a] * k[b] for a in 1:4, b in 1:4)

    # Symmetrize to cancel round-off errors
    g = (g + g') / 2

    return g::SMatrix{4,4}
end

################################################################################

# Kerr-Newman metric in fully harmonic coordinates
# (Cook, "Initial Data for Numerical Relativity", §3.3.2, eqs. 96–102.)
export Harmonic
"""
    Harmonic(M, a=0, Q=0)
    Harmonic{T}(M, a=0, Q=0)

Kerr-Newman black hole in **fully harmonic** Cartesian coordinates
`(t, x, y, z)`, satisfying `□xᵘ = 0` for all four coordinates.

Implemented via a smooth pullback from Kerr-Schild Cartesian coordinates
(closed-form coordinate map and analytic Jacobian), so the resulting metric is
regular on the z-axis as well as across the future event horizon. Valid for
`r > r₋ = M − √(M² − a² − Q²)`; singular on the disk `z = 0`, `x² + y² ≤ a²`.

Parameters: mass `M > 0`, spin and charge constrained by `a² + Q² < M²`.

Reference: Cook, *Initial Data for Numerical Relativity*, §3.3.2.
"""
struct Harmonic{T} <: AbstractMetric
    mass::T                     # 0 < M
    spin::T                     # M² > a² + Q²
    charge::T                   # Q
    Harmonic{T}(mass::T, spin::T, charge::T) where {T} = new{T}(mass, spin, charge)
end
function Harmonic{T}(mass, spin=zero(T), charge=zero(T)) where {T}
    Harmonic{T}(T(mass), T(spin), T(charge))
end
function Harmonic(mass, spin=0, charge=0)
    T = promote_type(typeof(mass), typeof(spin), typeof(charge))
    Harmonic{T}(mass, spin, charge)
end

Base.nameof(ha::Harmonic) = "Kerr-Newman metric in fully harmonic coordinates (M=$(ha.mass), a=$(ha.spin), Q=$(ha.charge))"

# Inner: Kerr-Newman in Kerr-Schild Cartesian coordinates.
# g = η + f k k with f = (2Mr - Q²)/ρ² and the null vector
#     k = (1, (r x + a y)/(r²+a²), (r y - a x)/(r²+a²), z/r).
# Regular on the z-axis (k has well-defined limits there).
function _kerr_newman_ks_metric(M, a, Q, p_ks::AbstractVector)
    @assert length(p_ks) == 4
    t, x, y, z = p_ks

    s = x^2 + y^2 + z^2 - a^2
    r = sqrt((s + sqrt(s^2 + 4 * a^2 * z^2)) / 2)

    ρ² = r^2 + a^2 * (z / r)^2
    f = (2M*r - Q^2) / ρ²
    rA = r^2 + a^2
    k = SVector(one(r), (r*x + a*y)/rA, (r*y - a*x)/rA, z/r)

    g = SMatrix{4,4}(η[i, j] + f * k[i] * k[j] for i in 1:4, j in 1:4)
    g = (g + g') / 2
    return g::SMatrix{4,4}
end

# Outer: harmonic Cartesian → Kerr-Schild Cartesian pullback.
#
# The Cook §3.3.2 spheroidal chart (used by Cook eqs. 96–102) has the usual
# r/θ/φ coordinate singularity on the z-axis. Instead, route through Kerr-Schild
# Cartesian, whose coordinate map and Jacobian are smooth everywhere away from
# the disk (R = 0) and the Cauchy horizon (r = r₋).
#
# Given harmonic Cartesian (t, x, y, z): solve the quartic for R = r − M, then
#     x_KS = (x (rR + a²) + M a y) / (R² + a²)
#     y_KS = (y (rR + a²) − M a x) / (R² + a²)
#     z_KS = r z / R
#     t_KS = t − 2M ln(2M / (r − r₋))
# All four expressions, and the analytic Jacobian below, are smooth on the
# z-axis (where x = y = 0 makes R_x = R_y = 0, x_KS = y_KS = 0).
function metric(ha::Harmonic, p::AbstractVector)
    M = ha.mass
    a = ha.spin
    Q = ha.charge

    @assert length(p) == 4
    t, x, y, z = p

    # Solve R⁴ − R²(x² + y² + z² − a²) − a²z² = 0 for R = r − M
    s = x^2 + y^2 + z^2 - a^2
    R² = (s + sqrt(s^2 + 4 * a^2 * z^2)) / 2
    R = sqrt(R²)
    r = R + M

    r₋ = M - sqrt(M^2 - a^2 - Q^2)

    A = R² + a^2
    B = r*R + a^2

    x_ks = (x*B + M*a*y) / A
    y_ks = (y*B - M*a*x) / A
    z_ks = r * z / R
    t_ks = t - 2M * log(2M / (r - r₋))

    g_ks = _kerr_newman_ks_metric(M, a, Q, SVector(t_ks, x_ks, y_ks, z_ks))

    # Implicit-derivative partials of R w.r.t. (x, y, z), from the quartic
    ξ = R²^2 + a^2 * z^2
    R_x = R² * R * x / ξ
    R_y = R² * R * y / ξ
    R_z = R * A * z / ξ

    fx = x * (2R + M) - 2R * x_ks
    fy = y * (2R + M) - 2R * y_ks

    Jtx = 2M * R_x / (r - r₋)
    Jty = 2M * R_y / (r - r₋)
    Jtz = 2M * R_z / (r - r₋)
    Jxx = (B + R_x * fx) / A
    Jxy = (M*a + R_y * fx) / A
    Jxz = (R_z * fx) / A
    Jyx = (-M*a + R_x * fy) / A
    Jyy = (B + R_y * fy) / A
    Jyz = (R_z * fy) / A
    Jzx = -z * M * R_x / R²
    Jzy = -z * M * R_y / R²
    Jzz = (1 + M/R) - z * M * R_z / R²

    o = one(R)
    ze = zero(R)
    # J[μ, ν] = ∂x_ks^μ / ∂x_h^ν, column-major (column = derivative variable).
    J = SMatrix{4,4}(o, ze, ze, ze, Jtx, Jxx, Jyx, Jzx, Jty, Jxy, Jyy, Jzy, Jtz, Jxz, Jyz, Jzz)

    g = J' * g_ks * J
    g = (g + g') / 2
    return g::SMatrix{4,4}
end

################################################################################

# @compile_workload begin
#     ks = KerrSchild(1.0, 0.3)
#     ha = Harmonic(1.0, 0.3)
#     x = SVector(0.0, 3.0, 1.0, 0.5)
#     metric(ks, x)
#     dmetric(ks, x)
#     ddmetric(ks, x)
#     ChristoffelSymbols(ks, x)
#     dChristoffelSymbols(ks, x)
#     RiemannTensor(ks, x)
#     RicciTensor(ks, x)
#     EinsteinTensor(ks, x)
#     metric(ha, x)
#     EinsteinTensor(ha, x)
# end

end
