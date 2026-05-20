module SpacetimeMetrics

using ForwardDiff
using LinearAlgebra
using PrecompileTools
using StaticArrays

export AbstractMetric
"""
    AbstractMetric

Supertype for analytic 4-dimensional Lorentzian spacetime metrics.

Conventions throughout the package:
- spacetime has four dimensions,
- metric signature is `(-1, +1, +1, +1)`,
- geometric units, `c = G = 1`.

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

    γ = g[2:4, 2:4]
    γu = inv(γ)
    βl = SVector{3}(g[2, 1], g[3, 1], g[4, 1])
    β = γu * βl
    α = sqrt(-g[1, 1] + dot(β, βl))

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

    # Γ^a_bc
    Γ = SArray{Tuple{4,4,4}}(sum(gu[a, x] * Γl[x, b, c] for x in 1:4) for a in 1:4, b in 1:4, c in 1:4)

    # Symmetrize to cancel round-off errors
    Γ = SArray{Tuple{4,4,4}}((Γ[a, b, c] + Γ[a, c, b]) / 2 for a in 1:4, b in 1:4, c in 1:4)

    return Γ::SArray{Tuple{4,4,4}}
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

    # Symmetrize to cancel round-off errors
    dΓ = SArray{Tuple{4,4,4,4}}((dΓ[a, b, c, d] + dΓ[a, c, b, d]) / 2 for a in 1:4, b in 1:4, c in 1:4, d in 1:4)

    return Γ::SArray{Tuple{4,4,4}}, dΓ::SArray{Tuple{4,4,4,4}}
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

    # R^a_bcd
    # TODO: Check the sign convention!
    Rm = SArray{Tuple{4,4,4,4}}(
        dΓ[a, d, b, c] - dΓ[a, c, b, d] + sum(Γ[a, c, x] * Γ[x, d, b] - Γ[a, d, x] * Γ[x, c, b] for x in 1:4) for
        a in 1:4, b in 1:4, c in 1:4, d in 1:4
    )

    # (Anti-)symmetrize to cancel round-off errors
    # We should probably apply more symmetries/antisymmetries
    Rm = SArray{Tuple{4,4,4,4}}((Rm[a, b, c, d] - Rm[a, b, d, c]) / 2 for a in 1:4, b in 1:4, c in 1:4, d in 1:4)

    return Rm::SArray{Tuple{4,4,4,4}}
end

export RicciTensor
"""
    RicciTensor(m::AbstractMetric, p::AbstractVector) -> Rc

Return the Ricci tensor `Rc[a, b] = R_{ab} = R^c_{acb}` at the 4-position `p`,
the trace of the Riemann tensor over the first and third indices. Symmetric.
"""
function RicciTensor(m::AbstractMetric, p::AbstractVector)
    Rm = RiemannTensor(m, p)

    # R_ab
    # TODO: Check the sign convention!
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

export Minkowski
"""
    Minkowski()

The flat Minkowski metric `η = diag(-1, +1, +1, +1)`.
"""
struct Minkowski <: AbstractMetric end

Base.nameof(::Minkowski) = "Minkowski metric"

metric(::Minkowski, ::AbstractVector{T}) where {T} = SMatrix{4,4,T}(η)

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
