abstract AbstractModel{CliqueSize}

# various dialects
typealias UnaryModel AbstractModel{1}
typealias DataCost AbstractModel{1}
typealias DataTerm AbstractModel{1}

typealias PairwiseModel AbstractModel{2}
typealias SmoothCost AbstractModel{2}
typealias SmoothTerm AbstractModel{2}
typealias RegularTerm AbstractModel{2}

typealias TreyModel AbstractModel{3}

typealias QuadraModel AbstractModel{4}


"""
    SAD()

The sum of absolute differences including variations.
"""
immutable SAD{F<:Function} <: DataCost
    f::F
end
SAD() = SAD(sadexp)

"""
    SSD()

The sum of squared differences including variations.
"""
immutable SSD{F<:Function} <: DataCost
    f::F
end
SSD() = SSD(ssdexp)


"""
    default_potts(𝓭, d) -> Vector{vals}

Returns cost value block calculated via `pottsexp`.
"""
default_potts(𝓭::AbstractVector, d) = [[pottsexp(α, β, d) for α in 𝓭, β in 𝓭]]

"""
    Potts()
    Potts(d)

The potts model. The default value of `d` is `1.0`.
"""
immutable Potts{F<:Function, T<:Real} <: SmoothCost
    f::F
    d::T
end
Potts(d=1.0) = Potts(default_potts, d)


"""
    default_tad(𝓭, c, d) -> Vector{vals}

Returns cost value block calculated via `tadexp`.
"""
default_tad(𝓭::AbstractVector, c, d) = [[tadexp(α, β, c, d) for α in 𝓭, β in 𝓭]]

"""
    TAD()
    TAD(c,d)
    TAD(d=10)

The truncated absolute difference.

# Arguments
* `c::Real=1.0`: the rate of increase in the cost.
* `d::Real=Inf`: controls when the cost stops increasing.
"""
immutable TAD{F<:Function,Tc<:Real,Td<:Real} <: SmoothCost
    f::F
    c::Tc
    d::Td
end
TAD(c,d) = TAD(default_tad, c, d)
TAD(;c=1.0, d=Inf) = TAD(c, d)


"""
    default_tqd(𝓭, c, d) -> Vector{vals}

Returns cost value block calculated via `tqdexp`.
"""
default_tqd(𝓭::AbstractVector, c, d) = [[tqdexp(α, β, c, d) for α in 𝓭, β in 𝓭]]

"""
    TQD()
    TQD(c,d)
    TQD(d=10)

The truncated quadratic difference.

# Arguments
* `c::Real=1.0`: the rate of increase in the cost.
* `d::Real=Inf`: controls when the cost stops increasing.
"""
immutable TQD{F<:Function,Tc<:Real,Td<:Real} <: SmoothCost
    f::F
    c::Tc
    d::Td
end
TQD(c,d) = TQD(default_tqd, c, d)
TQD(;c=1.0, d=Inf) = TQD(c, d)


"""
    topology2d(d) -> Vector{vals}

Returns 4 cost value blocks calculated from `jᶠᶠ`, `jᵇᶠ`, `jᶠᵇ`, `jᵇᵇ` respectively.
"""
@inline topology2d(d::AbstractVector) = [[jᶠᶠ(α, β, χ) for α in d, β in d, χ in d], [jᵇᶠ(α, β, χ) for α in d, β in d, χ in d],
                                         [jᶠᵇ(α, β, χ) for α in d, β in d, χ in d], [jᵇᵇ(α, β, χ) for α in d, β in d, χ in d]]

"""
    TP2D()

The topology preservation cost for 2D images(3-element cliques).
"""
immutable TP2D{F<:Function} <: TreyModel
    f::F
end
TP2D() = TP2D(topology2d)


"""
    topology3d(d) -> Vector{vals}

Returns 8 cost value blocks calculated from `jᶠᶠᶠ`, `jᵇᶠᶠ`, `jᶠᵇᶠ`, `jᵇᵇᶠ`,
`jᶠᶠᵇ`, `jᵇᶠᵇ`, `jᶠᵇᵇ`, `jᵇᵇᵇ` espectively.
"""
@inline topology3d(d::AbstractVector) = [[jᶠᶠᶠ(α, β, χ, δ) for α in d, β in d, χ in d, δ in d],
                                         [jᵇᶠᶠ(α, β, χ, δ) for α in d, β in d, χ in d, δ in d],
                                         [jᶠᵇᶠ(α, β, χ, δ) for α in d, β in d, χ in d, δ in d],
                                         [jᵇᵇᶠ(α, β, χ, δ) for α in d, β in d, χ in d, δ in d],
                                         [jᶠᶠᵇ(α, β, χ, δ) for α in d, β in d, χ in d, δ in d],
                                         [jᵇᶠᵇ(α, β, χ, δ) for α in d, β in d, χ in d, δ in d],
                                         [jᶠᵇᵇ(α, β, χ, δ) for α in d, β in d, χ in d, δ in d],
                                         [jᵇᵇᵇ(α, β, χ, δ) for α in d, β in d, χ in d, δ in d]]

"""
    TP3D()

The topology preservation cost for 3D images(4-element cliques).
"""
immutable TP3D{F<:Function} <: QuadraModel
    f::F
end
TP3D() = TP3D(topology3d)

# topology
typealias TopologyCost Union{TP2D, TP3D}
