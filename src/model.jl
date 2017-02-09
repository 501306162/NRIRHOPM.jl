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

# topology
abstract TopologyCost2D <: TreyModel
abstract TopologyCost3D <: QuadraModel
typealias TopologyCost Union{TopologyCost2D, TopologyCost3D}


# unary potentials
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


# pairwise potentials
"""
    Potts()
    Potts(d)

The potts model.

# Arguments
* `d::Real=1.0`: the constant value in Potts model.
"""
immutable Potts{F<:Function, T<:Real} <: SmoothCost
    f::F
    d::T
end
Potts() = Potts(potts, 1.0)
Potts(d) = Potts(potts, d)

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
TAD(c,d) = TAD(tad, c, d)
TAD(;c=1.0, d=Inf) = TAD(tad, c, d)

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
TQD(c,d) = TQD(tqd, c, d)
TQD(;c=1.0, d=Inf) = TQD(tqd, c, d)


# high-order potentials
"""
    TP2D()

The topology preservation cost for 2D images(3-element cliques).
"""
immutable TP2D{FF<:Function,BF<:Function,FB<:Function,BB<:Function} <: TopologyCost2D
    Jᶠᶠ::FF
    Jᵇᶠ::BF
    Jᶠᵇ::FB
    Jᵇᵇ::BB
end
TP2D() = TP2D(jᶠᶠ, jᵇᶠ, jᶠᵇ, jᵇᵇ)

"""
    TP3D()

The topology preservation cost for 3D images(4-element cliques).
"""
immutable TP3D{FFF<:Function,BFF<:Function,FBF<:Function,BBF<:Function,FFB<:Function,BFB<:Function,FBB<:Function,BBB<:Function} <: TopologyCost3D
    Jᶠᶠᶠ::FFF
    Jᵇᶠᶠ::BFF
    Jᶠᵇᶠ::FBF
    Jᵇᵇᶠ::BBF
    Jᶠᶠᵇ::FFB
    Jᵇᶠᵇ::BFB
    Jᶠᵇᵇ::FBB
    Jᵇᵇᵇ::BBB
end
TP3D() = TP3D(jᶠᶠᶠ, jᵇᶠᶠ, jᶠᵇᶠ, jᵇᵇᶠ, jᶠᶠᵇ, jᵇᶠᵇ, jᶠᵇᵇ, jᵇᵇᵇ)
