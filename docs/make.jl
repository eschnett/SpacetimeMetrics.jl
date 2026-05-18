# Generate documentation with this command:
# (cd docs && julia make.jl)

push!(LOAD_PATH, "..")

using Documenter
using SpacetimeMetrics

makedocs(; sitename="SpacetimeMetrics", format=Documenter.HTML(), modules=[SpacetimeMetrics])

deploydocs(; repo="github.com/eschnett/SpacetimeMetrics.jl.git", devbranch="main", push_preview=true)
