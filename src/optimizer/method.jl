abstract AbstractMethod
abstract AbstractHOPMMethod <: AbstractMethod


"""
    CanonHOPM()
    CanonHOPM(tolerance, maxIteration)
    CanonHOPM(tolerance=1e-6)

The canonical high order power method.
"""
immutable CanonHOPM{F<:Function,Tt<:Real,Tm<:Integer} <: AbstractHOPMMethod
    f::F
    tolerance::Tt
    maxIteration::Tm
end
CanonHOPM(tolerance, maxIteration) = CanonHOPM(hopm_canonical, tolerance, maxIteration)
CanonHOPM(;tolerance=1e-5, maxIteration=300) = CanonHOPM(tolerance, maxIteration)

"""
    MixHOPM()
    MixHOPM(constraint, tolerance, maxIteration)
    MixHOPM(constraint=:column)

The "mixed" high order power method.
"""
immutable MixHOPM{F<:Function,Tt<:Real,Tm<:Integer} <: AbstractHOPMMethod
    f::F
    constraint::Symbol
    tolerance::Tt
    maxIteration::Tm
end
MixHOPM(constraint, tolerance, maxIteration) = MixHOPM(hopm_mixed, constraint, tolerance, maxIteration)
MixHOPM(;constraint=:vecnorm, tolerance=1e-5, maxIteration=30) = MixHOPM(constraint, tolerance, maxIteration)
