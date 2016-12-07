using NRIRHOPM
using Base.Test

# construct a random example
fixedImg = rand(5,5)
movingImg = rand(size(fixedImg))
deformableWindow = [[i,j] for i in -1:1, j in -1:1]

# test for unaryclique
info("Testing unaryclique:")
𝐇¹ = unaryclique(fixedImg, movingImg, deformableWindow, SAD())
@test unaryclique(fixedImg, movingImg, deformableWindow) == 𝐇¹
@test unaryclique(fixedImg, movingImg, deformableWindow, algorithm=SAD()) == 𝐇¹

type Unknown <: DataCost
end
@test_throws ArgumentError unaryclique(fixedImg, movingImg, deformableWindow, algorithm=Unknown())
println("Passed.")

# test for pairwiseclique
info("Tesing pairwiseclique:")
imageDims = size(fixedImg)
deformers = reshape(deformableWindow, length(deformableWindow))
deformers = [tuple(v...) for v in deformers]
𝐇² = pairwiseclique(imageDims, deformers, TAD(), 1.0, 1.0, Inf);
𝐇²′ = pairwiseclique(fixedImg, movingImg, deformableWindow);
@test  𝐇²′.data == 𝐇².data
@test  𝐇²′.index == 𝐇².index
@test  𝐇²′.dims == 𝐇².dims

type Missing <: SmoothTerm
end
@test_throws ArgumentError pairwiseclique(fixedImg, movingImg, deformableWindow, algorithm=Missing())

println("Passed.")

# test for treyclique
info("Tesing treyclique:")
imageDims = size(fixedImg)
deformers = reshape(deformableWindow, length(deformableWindow))
𝐇³ = treyclique(imageDims, deformers, TP(), 1.0);
𝐇³′ = treyclique(fixedImg, movingImg, deformableWindow);
@test  𝐇³′.data == 𝐇³.data
@test  𝐇³′.index == 𝐇³.index
@test  𝐇³′.dims == 𝐇³.dims

type Fake <: TreyPotential
end
@test_throws ArgumentError treyclique(fixedImg, movingImg, deformableWindow, algorithm=Fake())

println("Passed.")
