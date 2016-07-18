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
    fixedImg::Array{T,N},
    movingImg::Array{T,N},
    deformableWindow::Matrix{Vector{Int}},
    metric::SAD;
    δ::Real=1e2
    )
    deformers = reshape(deformableWindow, length(deformableWindow))
    return sum_absolute_diff(fixedImg, movingImg, deformers, Float64(δ))
end

"""
Method for Mutual Information
"""
function unaryclique{T,N}(
    fixedImg::Array{T,N},
    movingImg::Array{T,N},
    deformableWindow::Matrix{Vector{Int}},
    metric::MI
    )
    deformers = reshape(deformableWindow, length(deformableWindow))
    return mutual_info(fixedImg, movingImg, deformers)
end

"""
Construct the first-order potential tensor(vector), also called **data cost** or **(dis)similarity measure**.
"""
function unaryclique{T,N}(
    fixedImg::Array{T,N},
    movingImg::Array{T,N},
    deformableWindow::Matrix{Vector{Int}};
    metric::AbstractDataCost = SAD(),
    δ::Real=1e2
    )
    # call corresponding methods
    info("Calling unaryclique:")
    if metric == SAD()
        info("Metric: SAD(Sum of Absolute Differences)")
        δ == 1e2 && info("You may need to specify the parameter δ when using SAD as metric. The default value is 100.")
        return unaryclique(fixedImg, movingImg, deformableWindow, metric; δ = δ)
    elseif metric == MI()
        info("Metric: MI(Mutual Information)")
        return unaryclique(fixedImg, movingImg, deformableWindow, metric)
    else
        error("The implementation of $(metric) is missing.")
    end
end
