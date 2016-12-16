# types for multi-dispatching
abstract AbstractPotential{Order}

# various dialects
typealias UnaryPotential AbstractPotential{1}
typealias DataTerm AbstractPotential{1}
typealias DataCost AbstractPotential{1}

typealias PairwisePotential AbstractPotential{2}
typealias SmoothTerm AbstractPotential{2}
typealias SmoothCost AbstractPotential{2}
typealias RegularTerm AbstractPotential{2}

typealias TreyPotential AbstractPotential{3}
typealias TopologyCost AbstractPotential{3}


# unary potentials
"""
    SAD()

The sum of absolute differences.
"""
type SAD <: DataCost
    𝓕::Function    # 𝓕 (\mbfscrF)
end
SAD() = SAD(sum_absolute_diff)

"""
    SSD()

The sum of squared differences.
"""
type SSD <: DataCost
    𝓕::Function    # 𝓕 (\mbfscrF)
end
SSD() = SSD(sum_squared_diff)


# pairwise potentials
"""
    Potts()
    Potts(d)

The potts model.

# Arguments
* `d::Real=1.0`: the constant value in Potts model.
"""
type Potts <: SmoothCost
    𝓕::Function    # 𝓕 (\mbfscrF)
    d::Real
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
type TAD <: SmoothCost
    𝓕::Function    # 𝓕 (\mbfscrF)
    c::Real
    d::Real
end
TAD(c,d) = TAD(truncated_absolute_diff, c, d)
TAD(;c=1.0, d=Inf) = TAD(truncated_absolute_diff, c, d)

"""
    TQD()
    TQD(c,d)
    TQD(d=10)

The truncated quadratic difference.

# Arguments
* `c::Real=1.0`: the rate of increase in the cost.
* `d::Real=Inf`: controls when the cost stops increasing.
"""
type TQD <: SmoothCost
    𝓕::Function    # 𝓕 (\mbfscrF)
    c::Real
    d::Real
end
TQD(c,d) = TQD(truncated_quadratic_diff, c, d)
TQD(;c=1.0, d=Inf) = TQD(truncated_quadratic_diff, c, d)


# high-order potentials
"""
    TP()

The topology preservation cost.
"""
type TP <: TopologyCost
    Jᶠᶠ::Function
    Jᵇᶠ::Function
    Jᶠᵇ::Function
    Jᵇᵇ::Function
end
TP() = TP(jᶠᶠ, jᵇᶠ, jᶠᵇ, jᵇᵇ)
