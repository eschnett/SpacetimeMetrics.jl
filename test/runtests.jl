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
