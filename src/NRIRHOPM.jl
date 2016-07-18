module NRIRHOPM
using StatsBase

# core
include("core.jl")

# data cost
include("datacost/data_cost.jl")
include("datacost/sum_absolute_diff.jl")
include("datacost/mutual_info.jl")

# regularization
include("regularization/regularization.jl")
include("regularization/truncated_absolute_diff.jl")

# integer programming
include("constraints.jl")

# core
export hopm, SharedSparseTensor, share, SparseArray

# data cost
export AbstractDataCost, SAD, MI, unaryclique

# regularization
export AbstractRegularization, Potts, TAD, Quadratic, pairwiseclique

# integer programming
export integerlize, integerhopm

end # module
