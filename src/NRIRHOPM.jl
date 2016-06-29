module NRIRHOPM

include("hopm.jl")
include("potentials.jl")
include("cliques.jl")
include("constraints.jl")


export hopm, SharedSparseTensor, share, SparseArray
export unaryclique, pairwiseclique, treyclique
export integerlize, integerhopm

end # module
