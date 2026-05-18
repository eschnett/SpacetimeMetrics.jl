module SpacetimeMetrics

using ForwardDiff
using LinearAlgebra
using PrecompileTools
using StaticArrays

export AbstractMetric
abstract type AbstractMetric end

# Conventions:
# - spacetime has four dimensions
# - metric signature is (-1, +1, +1, +1)
# - c = G = 1

export metric
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
function dmetric(m::AbstractMetric, p::AbstractVector)
    p = SVector{4}(p)
    g_dual = metric(m, make_dual(p, _DMetricTag))
    g = SMatrix{4,4}(ForwardDiff.value(g_dual[a, b]) for a in 1:4, b in 1:4)
    dg = SArray{Tuple{4,4,4}}(ForwardDiff.partials(g_dual[a, b], c) for a in 1:4, b in 1:4, c in 1:4)
    return g, dg
end

export ddmetric
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

export ChristoffelSymbols
function ChristoffelSymbols(m::AbstractMetric, p::AbstractVector)
    g, dg = dmetric(m, p)
    gu = inv(g)

    Γl = SArray{Tuple{4,4,4}}((dg[a, b, c] + dg[a, c, b] - dg[b, c, a]) / 2 for a in 1:4, b in 1:4, c in 1:4)

    # Γ^a_bc
    Γ = SArray{Tuple{4,4,4}}(sum(gu[a, x] * Γl[x, b, c] for x in 1:4) for a in 1:4, b in 1:4, c in 1:4)

    # Symmetrize to cancel round-off errors
    Γ = SArray{Tuple{4,4,4}}((Γ[a, b, c] + Γ[a, c, b]) / 2 for a in 1:4, b in 1:4, c in 1:4)

    return Γ::SArray{Tuple{4,4,4}}
end

export dChristoffelSymbols
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
function RicciTensor(m::AbstractMetric, p::AbstractVector)
    Rm = RiemannTensor(m, p)

    # R_ab
    # TODO: Check the sign convention!
    Rc = SArray{Tuple{4,4}}(sum(Rm[x,a,x,b] for x in 1:4) for a in 1:4, b in 1:4)

    # Symmetrize to cancel round-off errors
    Rc = (Rc + Rc') / 2

    return Rc::SMatrix{4,4}
end

export EinsteinTensor
function EinsteinTensor(m::AbstractMetric, p::AbstractVector)
    g = metric(m, p)
    gu = inv(g)
    Rc = RicciTensor(m, p)

    Rs = sum(Rc[x,y] * gu[x,y] for x in 1:4, y in 1:4)

    # G_ab
    G = SArray{Tuple{4,4}}(Rc[a,b] - 1//2 * Rs * g[a,b] for a in 1:4, b in 1:4)

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
function translate(metric::AbstractMetric, distance::AbstractVector)
    return TranslatedMetric(metric, SVector{4}(distance))
end

Base.nameof(tm::TranslatedMetric) = nameof(tm.metric) * ", translated by $(tm.distance)"

metric(tm::TranslatedMetric, x::AbstractVector) = metric(tm.metric, SVector{4}(x) - tm.distance)

################################################################################

export Minkowski
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

Base.nameof(kh::Harmonic) = "Kerr-Newman metric in fully harmonic coordinates (M=$(kh.mass), a=$(kh.spin), Q=$(kh.charge))"

# 4-metric in the spheroidal (t, r, θ, φ) chart used by Cook eqs. 96–102.
# Built from the ADM lapse, shift and 3-metric as g_tt = -α² + γ_ij β^i β^j,
# g_ti = γ_ij β^j, g_ij = γ_ij.
function _kerr_harmonic_spheroidal_metric(kh::Harmonic, p::AbstractVector)
    M = kh.mass
    a = kh.spin
    Q = kh.charge

    @assert length(p) == 4
    t, r, θ, φ = p
    cos²θ = cos(θ)^2
    sin²θ = sin(θ)^2

    rt = sqrt(M^2 - a^2 - Q^2)
    r₊ = M + rt
    r₋ = M - rt

    ρ² = r^2 + a^2 * cos²θ
    Hh = (r + r₊) / (r - r₋)
    Kh = 2M / (r - r₋)
    Φ = (2M*r - Q^2) / ρ²

    α⁻² = 1 + Φ * Hh + (r₊^2 + a^2) / ρ² * Kh
    α² = inv(α⁻²)

    βʳ = α² * (r₊^2 + a^2) / ρ²
    βᶲ = -α² * (a / ρ²) * Kh

    γ_rr = (2 - (1 - Φ) * Hh) * Hh
    γ_rφ = -(1 + Φ * Hh) * a * sin²θ
    γ_θθ = ρ²
    γ_φφ = (r^2 + a^2 + Φ * a^2 * sin²θ) * sin²θ

    g_tt = -α² + γ_rr * βʳ^2 + 2 * γ_rφ * βʳ * βᶲ + γ_φφ * βᶲ^2
    g_tr = γ_rr * βʳ + γ_rφ * βᶲ
    g_tφ = γ_rφ * βʳ + γ_φφ * βᶲ
    o = zero(g_tt)

    g = SMatrix{4,4}(
        g_tt, g_tr, o,    g_tφ,
        g_tr, γ_rr, o,    γ_rφ,
        o,    o,    γ_θθ, o,
        g_tφ, γ_rφ, o,    γ_φφ,
    )
    return g::SMatrix{4,4}
end

function metric(kh::Harmonic, p::AbstractVector)
    M = kh.mass
    a = kh.spin

    @assert length(p) == 4
    t, x, y, z = p

    # Solve the quartic R⁴ - R²(x² + y² + z² - a²) - a² z² = 0 for R² = (r-M)²
    s = x^2 + y^2 + z^2 - a^2
    R² = (s + sqrt(s^2 + 4 * a^2 * z^2)) / 2
    R = sqrt(R²)
    r = R + M

    # Reconstruct spheroidal angles via atan2 of (sin, cos) extracted from the
    # forward map. Differentiable and stable away from the z-axis / disk.
    cos_θ = z / R
    sin²θ = 1 - cos_θ^2
    sin_θ = sqrt(sin²θ)
    denom = (R² + a^2) * sin_θ
    cos_φ = (R * x + a * y) / denom
    sin_φ = (R * y - a * x) / denom
    θ = atan(sin_θ, cos_θ)
    φ = atan(sin_φ, cos_φ)

    g_sph = _kerr_harmonic_spheroidal_metric(kh, SVector(t, r, θ, φ))

    # Spatial Jacobian K[i, A] = ∂x^i / ∂X^A. With R = r - M, ∂R/∂r = 1.
    Kspat = SMatrix{3,3}(
        sin_θ * cos_φ,
        sin_θ * sin_φ,
        cos_θ,
        R*cos_θ*cos_φ - a*cos_θ*sin_φ,
        R*cos_θ*sin_φ + a*cos_θ*cos_φ,
        -R*sin_θ,
        -y,
        x,
        zero(x),
    )
    Jspat = inv(Kspat)          # Jspat[A, i] = ∂X^A / ∂x^i

    T = eltype(Jspat)
    o, ze = one(T), zero(T)
    J = SMatrix{4,4,T}(
        o,  ze,           ze,           ze,
        ze, Jspat[1, 1],  Jspat[2, 1],  Jspat[3, 1],
        ze, Jspat[1, 2],  Jspat[2, 2],  Jspat[3, 2],
        ze, Jspat[1, 3],  Jspat[2, 3],  Jspat[3, 3],
    )

    g = J' * g_sph * J

    # Symmetrize to cancel round-off errors
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
