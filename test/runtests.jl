using LinearAlgebra
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
            @test isapprox(G, zero(G); atol=1e-8)
        end
    end
end

@testset "Translations" begin
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

@testset "Derivatives" begin
    for iter in 1:10
        M = 0.5 + rand()
        a = rand() - 0.5
        Q = 0
        ks = KerrSchild(M, a, Q)

        # Avoid testing near the singularity
        x = 2 .+ SVector{4}(randn(4))

        # Calculate metric and derivatives; test symmetries
        g = metric(ks, x)
        @test all(g[a, b] == g[b, a] for a in 1:4, b in 1:4)

        dg = dmetric(ks, x)
        @test all(dg[a, b, c] == dg[b, a, c] for a in 1:4, b in 1:4, c in 1:4)

        ddg = ddmetric(ks, x)
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
