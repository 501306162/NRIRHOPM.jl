"""
Calculates the truncated absolute difference between two transform vectors.
Returns the cost value.

Refer to the following paper for further details:

Felzenszwalb, Pedro F., and Daniel P. Huttenlocher. "Efficient belief propagation for early vision." International journal of computer vision 70.1 (2006): 41-54.
"""
function truncated_absolute_diff(
    fp::Vector{Int},             # transform vector at pixel p
    fq::Vector{Int};             # transform vector at pixel q
    c::Float64=1,                # the rate of increase in the cost
    d::Float64=Inf               # controls when the cost stops increasing
    )
    return min(c * abs(vecnorm(fp) - vecnorm(fq)), d)
end
