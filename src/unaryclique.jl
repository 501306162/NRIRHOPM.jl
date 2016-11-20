"""
    unaryclique(fixedImg, movingImg, deformableWindow; <keyword arguments>)

Construct the first-order potential tensor, also called **data term** or **(dis)similarity measure**.

# Arguments
* `fixedImg::Array{T,N}`: the fixed(target) image.
* `movingImg::Array{T,N}`: the moving(source) image.
* `deformableWindow::Matrix{Vector{Int}}`: the transform matrix.
* `algorithm::DataCost=SAD()`: the method for calculating data cost.
* `Δ::Real=1e3`: reserved parameter.

The default metric is SAD(Sum of Absolute Differences).
"""
function unaryclique{T,N}(
    fixedImg::Array{T,N},
    movingImg::Array{T,N},
    deformableWindow::Matrix{Vector{Int}};
    algorithm::DataCost=SAD(),
    Δ::Real=1e3
    )
    info("Calling unaryclique:")
    if algorithm == SAD()
        info("Algorithm: SAD(Sum of Absolute Differences)")
        return unaryclique(fixedImg, movingImg, deformableWindow, algorithm)
    else
        throw(ArgumentError("The implementation of $(algorithm) is missing."))
    end
end

"""
    unaryclique(fixedImg, movingImg, deformableWindow, datacost) -> Vector

The method for the Sum of Absolute Differences(SAD). Returns a `Vector` 𝐇¹.
"""
function unaryclique{T,N}(
    fixedImg::Array{T,N},
    movingImg::Array{T,N},
    deformableWindow::Matrix{Vector{Int}},
    datacost::SAD
    )
    deformers = reshape(deformableWindow, length(deformableWindow))
    return sum_absolute_diff(fixedImg, movingImg, deformers)
end
