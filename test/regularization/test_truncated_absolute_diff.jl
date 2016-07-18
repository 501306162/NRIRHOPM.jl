# Test truncated_absolute_diff.jl
using Base.Test
# load source code
srcpath = realpath(joinpath(dirname(@__FILE__), "../../src/regularization/truncated_absolute_diff.jl"))
include(srcpath)

# construct two transform vectors
fp = rand(collect(1:1000), 2)
fq = rand(collect(1:1000), 2)

# call function
cost = truncated_absolute_diff(fp, fq; c=1.0, d=Inf)
delta = abs(sqrt(fp[1]^2 + fp[2]^2) - sqrt(fq[1]^2 + fq[2]^2))
@test cost == delta

# c = 1.0
rate = rand(1)[]
@test truncated_absolute_diff(fp, fq; c=rate, d=Inf) == rate * delta

# d = 0
@test truncated_absolute_diff(fp, fq; c=2.0, d=0.0) == 0
