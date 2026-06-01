using LinearAlgebra
using Random
using SpacetimeMetrics
using StaticArrays
using Test

function metrics_are_equal(m1, m2)
    for iter in 1:10
        x = randn(4)
        g1 = metric(m1, x)
        g2 = metric(m2, x)
        isapprox(g1, g2) || return false
    end
    return true
end

@testset "Minkowski" begin
    Random.seed!(1)
    minkowski = Minkowski()

    for iter in 1:10
        minkowski = Minkowski()
        x = randn(4)
        g = metric(minkowski, x)
        @test isdiag(g)
        @test diag(g) == [-1, +1, +1, +1]
    end
end

@testset "Kerr-Schild" begin
    Random.seed!(2)
    kerrschild = KerrSchild(1.0)
    kerrschild = KerrSchild(1.0, 0.5)
    kerrschild = KerrSchild(1.0, 0.5, 0.5)
    kerrschild = KerrSchild{Float64}(1.0)
    kerrschild = KerrSchild{Float64}(1.0, 0.5)
    kerrschild = KerrSchild{Float64}(1.0, 0.5, 0.5)

    for iter in 1:10
        M = 0.5 + rand()
        a = rand() - 0.5
        Q = 0
        ks = KerrSchild(M, a, Q)

        for n in 1:10
            x = randn(4)
            g = metric(ks, x)
            @test issymmetric(g)

            G = EinsteinTensor(ks, x)
            @test isapprox(G, zero(G); atol=1e-7)
        end
    end
end

@testset "Harmonic" begin
    Random.seed!(3)
    ha = Harmonic(1.0)
    ha = Harmonic(1.0, 0.5)
    ha = Harmonic(1.0, 0.5, 0.0)
    ha = Harmonic{Float64}(1.0)
    ha = Harmonic{Float64}(1.0, 0.5)
    ha = Harmonic{Float64}(1.0, 0.5, 0.0)

    for iter in 1:10
        M = 0.5 + rand()
        a = (rand() - 0.5) * 0.5
        Q = 0
        ha = Harmonic(M, a, Q)

        for n in 1:10
            x = 2 .+ SVector{4}(randn(4))
            g = metric(ha, x)
            @test issymmetric(g)

            G = EinsteinTensor(ha, x)
            @test isapprox(G, zero(G); atol=1e-8)
        end
    end

    # Electrovac: Q ≠ 0 sources an EM stress-energy. The full Einstein tensor is
    # nonzero, but the EM stress-energy is traceless, so the Ricci scalar
    # R = g^{ab} R_{ab} still vanishes.
    for iter in 1:10
        M = 0.5 + rand()
        a = (rand() - 0.5) * 0.3
        Q = (rand() - 0.5) * 0.3
        ha = Harmonic(M, a, Q)

        for n in 1:10
            x = 2 .+ SVector{4}(randn(4))
            g = metric(ha, x)
            @test issymmetric(g)

            gu = inv(g)
            Rc = RicciTensor(ha, x)
            Rs = sum(gu[a, b] * Rc[a, b] for a in 1:4, b in 1:4)
            @test isapprox(Rs, 0; atol=1e-8)
        end
    end

    # The harmonic chart and its derivatives are regular on the z-axis (away
    # from the disk z = 0, x² + y² ≤ a²). Test directly at x = y = 0.
    for iter in 1:10
        M = 0.5 + rand()
        a = (rand() - 0.5) * 0.5
        ha = Harmonic(M, a)

        for n in 1:10
            z = sign(randn()) * (1 + 2*rand())
            x = SVector(randn(), 0.0, 0.0, z)
            g = metric(ha, x)
            @test all(isfinite, g)
            @test issymmetric(g)

            G = EinsteinTensor(ha, x)
            @test all(isfinite, G)
            @test isapprox(G, zero(G); atol=1e-8)
        end
    end
end

@testset "Translations" begin
    Random.seed!(4)
    ks = KerrSchild(1.0, 0.5)

    for iter in 1:10
        dist = randn(4)
        tm = translate(ks, dist)

        for n in 1:10
            x = randn(4)
            g = metric(tm, x)
            @test g == metric(ks, x - dist)
        end

        tm2 = translate(tm, -dist)
        @test metrics_are_equal(ks, tm2)
    end
end

@testset "Rotations" begin
    Random.seed!(7)

    # Identity rotation is the identity transformation.
    let ks = KerrSchild(1.0, 0.3)
        rm = rotate(ks, 0.0, 0.0, 0.0)
        for n in 1:10
            x = 2 .+ SVector{4}(randn(4))
            @test isapprox(metric(rm, x), metric(ks, x))
        end
    end

    # Minkowski is rotation-invariant.
    let mink = Minkowski()
        for n in 1:10
            ψ, θ, φ = 2π*rand(), π*rand(), 2π*rand()
            rm = rotate(mink, ψ, θ, φ)
            x = randn(4)
            @test isapprox(metric(rm, x), metric(mink, x))
        end
    end

    # Schwarzschild (a = 0) is spherically symmetric: rotation leaves g invariant.
    for iter in 1:10
        M = 0.5 + rand()
        ks = KerrSchild(M)
        ψ, θ, φ = 2π*rand(), π*rand(), 2π*rand()
        rm = rotate(ks, ψ, θ, φ)
        for n in 1:10
            x = 2 .+ SVector{4}(randn(4))
            @test isapprox(metric(rm, x), metric(ks, x); atol=1e-12)
        end
    end

    # Rotation is a diffeomorphism: Einstein tensor still vanishes in vacuum.
    for iter in 1:10
        M = 0.5 + rand()
        a = (rand() - 0.5) * 0.5
        ks = KerrSchild(M, a)
        ψ, θ, φ = 2π*rand(), π*rand(), 2π*rand()
        rm = rotate(ks, ψ, θ, φ)
        for n in 1:10
            x = 2 .+ SVector{4}(randn(4))
            g = metric(rm, x)
            @test issymmetric(g)
            G = EinsteinTensor(rm, x)
            @test isapprox(G, zero(G); atol=1e-7)
        end
    end

    # Rotating Kerr by (0, π/2, 0) tilts the spin axis from ẑ to x̂:
    # the rotated metric at (t, x, y, z) should equal the original at (t, -z, y, x)
    # (since R_y(π/2) sends (x,y,z) ↦ (z, y, -x), so R_y(π/2)^T sends (x,y,z) ↦ (-z, y, x)).
    for iter in 1:10
        M = 0.5 + rand()
        a = (rand() - 0.5) * 0.5
        ks = KerrSchild(M, a)
        rm = rotate(ks, 0.0, π/2, 0.0)

        for n in 1:10
            x = 2 .+ SVector{4}(randn(4))
            x_old = SVector(x[1], -x[4], x[3], x[2])

            R₃ = SMatrix{3,3}(0.0, 0.0, -1.0,  0.0, 1.0, 0.0,  1.0, 0.0, 0.0)  # R_y(π/2), column-major
            R₄ = SMatrix{4,4}(1.0, 0.0, 0.0, 0.0,
                              0.0, R₃[1,1], R₃[2,1], R₃[3,1],
                              0.0, R₃[1,2], R₃[2,2], R₃[3,2],
                              0.0, R₃[1,3], R₃[2,3], R₃[3,3])
            @test isapprox(metric(rm, x), R₄ * metric(ks, x_old) * R₄'; atol=1e-12)
        end
    end
end

@testset "Boosts" begin
    Random.seed!(9)

    # |v| ≥ 1 is rejected.
    @test_throws ArgumentError boost(Minkowski(), [1.0, 0.0, 0.0])
    @test_throws ArgumentError boost(Minkowski(), [0.8, 0.8, 0.0])

    # Zero velocity is the identity transformation.
    let ks = KerrSchild(1.0, 0.3)
        bm = boost(ks, [0.0, 0.0, 0.0])
        for n in 1:10
            x = 2 .+ SVector{4}(randn(4))
            @test isapprox(metric(bm, x), metric(ks, x))
        end
    end

    # Minkowski is boost-invariant.
    let mink = Minkowski()
        for n in 1:10
            v = (rand(3) .- 0.5) * 0.9
            bm = boost(mink, v)
            x = randn(4)
            @test isapprox(metric(bm, x), metric(mink, x); atol=1e-12)
        end
    end

    # Boost is a Lorentz transformation: Einstein tensor still vanishes in vacuum.
    for iter in 1:10
        M = 0.5 + rand()
        a = (rand() - 0.5) * 0.5
        ks = KerrSchild(M, a)
        v = normalize(randn(3)) * (0.7 * rand())
        bm = boost(ks, v)
        for n in 1:10
            x = 2 .+ SVector{4}(randn(4))
            g = metric(bm, x)
            @test issymmetric(g)
            G = EinsteinTensor(bm, x)
            @test isapprox(G, zero(G); atol=1e-7)
        end
    end

    # Explicit pullback check against a hand-built x-boost matrix.
    for iter in 1:10
        M = 0.5 + rand()
        a = (rand() - 0.5) * 0.5
        ks = KerrSchild(M, a)
        β = 0.8 * rand()
        bm = boost(ks, [β, 0.0, 0.0])
        γ = 1 / sqrt(1 - β^2)
        Λ = SMatrix{4,4}(γ, γ*β, 0.0, 0.0,
                         γ*β, γ, 0.0, 0.0,
                         0.0, 0.0, 1.0, 0.0,
                         0.0, 0.0, 0.0, 1.0)
        for n in 1:10
            x = 2 .+ SVector{4}(randn(4))
            @test isapprox(metric(bm, x), Λ * metric(ks, Λ' * x) * Λ'; atol=1e-12)
        end
    end
end

@testset "GaugeWave" begin
    Random.seed!(8)

    H(A, d, t, x) = 1 - A * sin(2π * (x - t) / d)

    # Input validation
    @test_throws ArgumentError gaugewave(Minkowski(), 1.0, 1.0)
    @test_throws ArgumentError gaugewave(Minkowski(), -1.5, 1.0)
    @test_throws ArgumentError gaugewave(Minkowski(), 0.5, 0.0)
    @test_throws ArgumentError gaugewave(Minkowski(), 0.5, -1.0)

    # Closed-form match: g = diag(-H, H, 1, 1)
    for iter in 1:10
        A = (rand() - 0.5)        # |A| < 0.5
        d = 0.5 + rand()
        gw = GaugeWave(A, d)
        for n in 1:10
            x = randn(4)
            t, X = x[1], x[2]
            h = H(A, d, t, X)
            g = metric(gw, x)
            @test issymmetric(g)
            @test isapprox(g, diagm([-h, h, 1.0, 1.0]); atol=1e-12)
            @test h > 0          # signature preserved for |A| < 1
        end
    end

    # A = 0 reduces to Minkowski.
    let mink = Minkowski()
        @test metrics_are_equal(mink, GaugeWave(0.0, 1.0))
    end

    # The gauge wave is a diffeomorphism of flat space: curvature vanishes.
    for iter in 1:10
        A = (rand() - 0.5)
        d = 0.5 + rand()
        gw = GaugeWave(A, d)
        for n in 1:10
            x = randn(4)
            G = EinsteinTensor(gw, x)
            @test isapprox(G, zero(G); atol=1e-9)
            Rm = RiemannTensor(gw, x)
            @test isapprox(Rm, zero(Rm); atol=1e-9)
        end
    end

    # Genuinely time-dependent: ∂_t g ≠ 0 and K ≠ 0 (unlike static charts).
    for iter in 1:10
        A = 0.1 + 0.4 * rand()    # bounded away from 0
        d = 0.5 + rand()
        gw = GaugeWave(A, d)
        x = randn(4)
        _, dg = dmetric(gw, x)
        @test maximum(abs, dg[:, :, 1]) > 1e-6
        K = ExtrinsicCurvature(gw, x)
        @test issymmetric(K)
        @test maximum(abs, K) > 1e-6
    end
end

@testset "ExtrinsicCurvature" begin
    Random.seed!(6)

    # Minkowski: constant-t slices of flat space are flat — K = 0.
    minkowski = Minkowski()
    for n in 1:10
        x = randn(4)
        K = ExtrinsicCurvature(minkowski, x)
        @test isapprox(K, zero(K); atol=1e-12)
    end

    # ADM identity:  K_{ij} = -(1/(2α)) (∂_t γ_{ij} − L_β γ_{ij})
    # with  L_β γ_{ij} = β^k ∂_k γ_{ij} + γ_{kj} ∂_i β^k + γ_{ik} ∂_j β^k.
    # Compare ExtrinsicCurvature to a finite-difference evaluation of the RHS.
    e4(μ) = SVector{4}(ntuple(i -> i == μ ? 1.0 : 0.0, 4))
    for iter in 1:10
        M = 0.5 + rand()
        a = (rand() - 0.5) * 0.5
        ks = KerrSchild(M, a)

        for n in 1:10
            x = 2 .+ SVector{4}(randn(4))
            K = ExtrinsicCurvature(ks, x)

            g = metric(ks, x)
            γ = g[2:4, 2:4]
            βl = SVector{3}(g[2, 1], g[3, 1], g[4, 1])
            β = inv(γ) * βl
            α = sqrt(-g[1, 1] + dot(β, βl))

            shift_up(y) = let gy = metric(ks, y)
                inv(gy[2:4, 2:4]) * SVector{3}(gy[2, 1], gy[3, 1], gy[4, 1])
            end

            ε = 1e-5
            dμ_γ = ntuple(μ -> (metric(ks, x + ε*e4(μ))[2:4, 2:4] - metric(ks, x - ε*e4(μ))[2:4, 2:4]) / (2ε), 4)
            dk_β = ntuple(k -> (shift_up(x + ε*e4(k + 1)) - shift_up(x - ε*e4(k + 1))) / (2ε), 3)

            Lβγ = SMatrix{3,3}(
                sum(β[k] * dμ_γ[k + 1][i, j] for k in 1:3)
                + sum(γ[k, j] * dk_β[i][k] for k in 1:3)
                + sum(γ[i, k] * dk_β[j][k] for k in 1:3)
                for i in 1:3, j in 1:3
            )

            K_fd = -(dμ_γ[1] - Lβγ) / (2α)
            @test isapprox(K, K_fd; atol=1e-6)
            @test issymmetric(K)
        end
    end

    # Analytic K for Schwarzschild in Kerr-Schild Cartesian (a = Q = 0):
    #   f = 2M/r,  k_i = x_i/r,  α = 1/√(1+f),  ∂_t γ_{ij} = 0
    #   K_{ij} = (2M / (r² √(1 + 2M/r))) · [δ_{ij} − (2 + M/r) k_i k_j]
    for iter in 1:10
        M = 0.5 + rand()
        ks = KerrSchild(M)

        for n in 1:10
            # Stay safely away from the singularity: pick r ∈ [1M, 4M].
            n̂ = normalize(SVector{3}(randn(3)))
            r = (1 + 3*rand()) * M
            xyz = r * n̂
            x = SVector(randn(), xyz[1], xyz[2], xyz[3])
            k = n̂

            pref = 2M / (r^2 * sqrt(1 + 2M/r))
            K_analytic = SMatrix{3,3}(
                pref * ((i == j ? 1.0 : 0.0) - (2 + M/r) * k[i] * k[j]) for i in 1:3, j in 1:3
            )

            K = ExtrinsicCurvature(ks, x)
            @test isapprox(K, K_analytic; atol=1e-12)
        end
    end
end

@testset "Derivatives" begin
    Random.seed!(5)
    for iter in 1:10
        M = 0.5 + rand()
        a = rand() - 0.5
        Q = 0
        ks = KerrSchild(M, a, Q)

        # Avoid testing near the singularity
        x = 2 .+ SVector{4}(randn(4))

        # Calculate metric and derivatives; test symmetries
        g, dg = dmetric(ks, x)
        @test all(g[a, b] == g[b, a] for a in 1:4, b in 1:4)
        @test all(dg[a, b, c] == dg[b, a, c] for a in 1:4, b in 1:4, c in 1:4)

        _, _, ddg = ddmetric(ks, x)
        @test all(ddg[a, b, c, d] == ddg[b, a, c, d] for a in 1:4, b in 1:4, c in 1:4, d in 1:4)
        @test all(ddg[a, b, c, d] == ddg[a, b, d, c] for a in 1:4, b in 1:4, c in 1:4, d in 1:4)

        # Compare to finite differences
        for n in 1:10
            dx = SVector{4}(randn(4))

            # Exact derivative
            dg0 = SMatrix{4,4}(sum(dg[a, b, c] * dx[c] for c in 1:4) for a in 1:4, b in 1:4)

            # Finite difference derivative
            epsilon = 1.0e-5
            gp = metric(ks, x + epsilon * dx)
            gm = metric(ks, x - epsilon * dx)
            dg1 = (gp - gm) / (2 * epsilon)

            @test isapprox(dg0, dg1)
        end
    end
end
