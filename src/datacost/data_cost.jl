"""
Root abstract type for multi-dispatching data cost terms
"""
abstract AbstractDataCost

"""
Sum of Absolute Differences
"""
type SAD <: AbstractDataCost
end

"""
Mutual Information
"""
type MI <: AbstractDataCost
end

"""
Method for Sum of Absolute Differences
"""
function unaryclique{T,N}(
    fixedImg::Array{T,N},                       # fixed(target) image
    movingImg::Array{T,N},                      # moving(source) image
    deformableWindow::Matrix{Vector{Int}},      # transform matrix
    metric::SAD                                 # Sum of Absolute Differences
    )
    deformers = reshape(deformableWindow, length(deformableWindow))
    return sum_absolute_diff(fixedImg, movingImg, deformers)
end

"""
Method for Mutual Information
"""
function unaryclique{T,N}(
    fixedImg::Array{T,N},                       # fixed(target) image
    movingImg::Array{T,N},                      # moving(source) image
    deformableWindow::Matrix{Vector{Int}},      # transform matrix
    metric::MI;                                 # Mutual Information
    β::Int=0                                   # number of bins used for histogram
    )
    if β == 0
        β = floor(sqrt(length(fixedImg)/5))
    end
    deformers = reshape(deformableWindow, length(deformableWindow))
    return mutual_info(fixedImg, movingImg, deformers, Int(β))
end

"""
Construct the first-order potential tensor(vector), also called **data term** or **(dis)similarity measure**.

Requires arguments:

- fixedImg::Array{T,N}                     # fixed(target) image
- movingImg::Array{T,N}                    # moving(source) image
- deformableWindow::Matrix{Vector{Int}}    # transform matrix
- metric::AbstractDataCost                 # keyword argument for metric selection
- β::Int                                   # number of bins used for histogram, argument for MI

The default metric is SAD(Sum of Absolute Differences).
"""
function unaryclique{T,N}(
    fixedImg::Array{T,N},                       # fixed(target) image
    movingImg::Array{T,N},                      # moving(source) image
    deformableWindow::Matrix{Vector{Int}};      # transform matrix
    metric::AbstractDataCost=SAD(),             # metric selection
    β::Int=0                                    # [MI] number of bins used for histogram
    )
    # call corresponding methods
    info("Calling unaryclique:")
    if metric == SAD()
        info("Metric: SAD(Sum of Absolute Differences)")
        return unaryclique(fixedImg, movingImg, deformableWindow, metric)
    elseif metric == MI()
        info("Metric: MI(Mutual Information)")
        β == 0 && warn("The parameter β was not explicitly specified. The default value is used. Note that, in some cases, this cannot guarantee a sane registration.")
        return unaryclique(fixedImg, movingImg, deformableWindow, metric; β=β)
    else
        error("The implementation of $(metric) is missing.")
    end
end
