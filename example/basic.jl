using NRIRHOPM, Interpolations
#  1  4  7        111  121  131
#  2  5  8   <=>  211  221  231
#  3  6  9        311  321  331
#-----------front--------------
# 10 13 16        112  122  132
# 11 14 17   <=>  212  222  232
# 12 15 18        312  322  332
#-----------middle-------------
# 19 22 25        113  123  133
# 20 23 26   <=>  213  223  233
# 21 24 27        313  323  333
#-----------back---------------
fixed = reshape([1:27;], 3, 3, 3)
moving = copy(fixed)

moving[1,3,2] = 14
moving[2,2,2] = 23
moving[2,2,3] = 25
moving[1,3,3] = 16

displacements = [SVector(i,j,k) for i in -1:1, j in -1:1, k in -1:1]

warpped, d, spec, energy = multilevel(fixed, moving, [displacements], [(3,3,3)], topology=TP3D(), β=0.1, χ=0.01)

topologyΔ = vecnorm(warpped[end]-fixed)
originΔ = vecnorm(moving-fixed)

# multilevel
dim0 = size(fixed)
dim1 = (6,6,6)
knots = ntuple(x->linspace(1, dim1[x], dim0[x]), Val{3})
fixedITP = interpolate(knots, fixed, Gridded(Linear()))
fixedUp = fixedITP[1:6,1:6,1:6]
movingITP = interpolate(knots, moving, Gridded(Linear()))
movingUp = movingITP[1:6,1:6,1:6]

displacements = [SVector(i,j,k) for i in -1:1, j in -1:1, k in -1:1]
displacementSet = [displacements, displacements]
gridSet = [dim0, dim1]

warpped, d, spec, energy = multilevel(fixedUp, movingUp, displacementSet, gridSet, topology=TP3D(), β=0.1, χ=0.01)

topologyΔ = vecnorm(warpped[end]-fixedUp)
originΔ = vecnorm(movingUp-fixedUp)
