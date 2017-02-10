abstract AbstractMethod
abstract AbstractHOPMMethod <: AbstractMethod

immutable MixHOPM{F<:Function,Tt<:Real,Tm<:Integer} <: AbstractHOPMMethod
    f::F
    constraint::Symbol
    tolerance::Tt
    maxIteration::Tm
end
MixHOPM() =
