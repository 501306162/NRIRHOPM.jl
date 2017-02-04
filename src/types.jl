# types for multi-dispatching
abstract AbstractPotential{Order,Dim}

# various dialects
typealias UnaryPotential AbstractPotential{1}
typealias DataTerm AbstractPotential{1}
typealias DataCost AbstractPotential{1}

typealias PairwisePotential AbstractPotential{2}
typealias SmoothTerm AbstractPotential{2}
typealias SmoothCost AbstractPotential{2}
typealias RegularTerm AbstractPotential{2}

typealias TreyPotential AbstractPotential{3}
typealias TopologyCost2D AbstractPotential{3,2}

typealias QuadraPotential AbstractPotential{4}
typealias TopologyCost3D AbstractPotential{4,3}

typealias TopologyCost Union{TopologyCost2D, TopologyCost3D}

# unary potentials
"""
    SAD()

The sum of absolute differences.
"""
immutable SAD{F<:Function} <: DataCost
    𝓕::F           # 𝓕 (\mbfscrF)
end
SAD() = SAD(sadexp)

"""
    SSD()

The sum of squared differences.
"""
immutable SSD{F<:Function} <: DataCost
    𝓕::F           # 𝓕 (\mbfscrF)
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
    𝓕::F           # 𝓕 (\mbfscrF)
    d::T
end
Potts() = Potts(potts_model, 1.0)
Potts(d) = Potts(potts_model, d)

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
    𝓕::F           # 𝓕 (\mbfscrF)
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
    𝓕::F           # 𝓕 (\mbfscrF)
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
immutable TP2D <: TopologyCost2D
    Jᶠᶠ::Function
    Jᵇᶠ::Function
    Jᶠᵇ::Function
    Jᵇᵇ::Function
end
TP2D() = TP2D(jᶠᶠ, jᵇᶠ, jᶠᵇ, jᵇᵇ)

"""
    TP3D()

The topology preservation cost for 3D images(4-element cliques).
"""
immutable TP3D <: TopologyCost3D
    Jᶠᶠᶠ::Function
    Jᵇᶠᶠ::Function
    Jᶠᵇᶠ::Function
    Jᵇᵇᶠ::Function
    Jᶠᶠᵇ::Function
    Jᵇᶠᵇ::Function
    Jᶠᵇᵇ::Function
    Jᵇᵇᵇ::Function
end
TP3D() = TP3D(jᶠᶠᶠ, jᵇᶠᶠ, jᶠᵇᶠ, jᵇᵇᶠ, jᶠᶠᵇ, jᵇᶠᵇ, jᶠᵇᵇ, jᵇᵇᵇ)
