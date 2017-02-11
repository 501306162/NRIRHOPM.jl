using TensorDecompositions
import NRIRHOPM: hopm_mixed, hopm_canonical
# for i = 1:100
@testset "hopm" begin
    tolerance = 1e-7
    maxIteration = 300
    valN = 3
    idxN = 4

    # construct pairwise tensor
    pvals = [ValueBlock(rand(valN, valN))]
    pindex = NTuple{2,Int}[]
    for 𝒊 in CartesianRange((idxN,idxN))
        if 𝒊[1] < 𝒊[2]
            push!(pindex, 𝒊.I)
        end
    end
    pidxs = [IndexBlock(pindex)]
    pdims = (valN, idxN, valN, idxN)
    𝐓 = BlockedTensor(pvals, pidxs, pdims)
    full𝐓 = full(𝐓)

    # construct high order tensor
    tvals = [ValueBlock(rand(valN, valN, valN))]
    tindex = NTuple{3,Int}[]
    for 𝒊 in CartesianRange((idxN,idxN,idxN))
        if 𝒊[1] < 𝒊[2] < 𝒊[3]
            push!(tindex, 𝒊.I)
        end
    end
    tidxs = [IndexBlock(tindex)]
    tdims = (valN, idxN, valN, idxN, valN, idxN)
    𝑻 = BlockedTensor(tvals, tidxs, tdims)
    full𝑻 = full(𝑻)

    @testset "2nd order canonical" begin
        A = reshape(full𝐓, valN*idxN, valN*idxN)
        𝐭 = rand(valN, idxN)
        A .+= diagm(reshape(𝐭, length(𝐭)))
        # build-in eigenpair function
        d, v, nconv, niter, nmult, resid = eigs(A, nev=1, which=:LM, tol=tolerance)
        v = abs.(v[:])

        # TensorDecompositions's dense method
        lbd, x = sshopm(A, 0, tol=tolerance, maxiter=maxIteration)
        x = abs.(x)

        # TensorDecompositions's sparse method
        As = SparseArray(A)
        lbdsparse, y = sshopm(As, 0, tol=tolerance, maxiter=maxIteration)
        y = abs.(y)

        # hopm_canonical
        energy, Z = hopm_canonical(𝐭, 𝐓, rand(valN, idxN), tolerance, maxIteration)
        z = reshape(Z, length(Z))

        @test d[] ≈ lbd ≈ lbdsparse ≈ energy
        @test vecnorm(z - x) < 5tolerance
        @test vecnorm(z - y) < 5tolerance
        @test vecnorm(z - v) < 5tolerance
    end

    @testset "3nd order canonical" begin
        A = reshape(full𝑻, valN*idxN, valN*idxN, valN*idxN)
        𝐭 = 0.1*rand(valN, idxN)    # the magnitude of 𝐭 should be smaller, need further investigation.
        for 𝒊 in CartesianRange(size(A))
            if 𝒊[1] == 𝒊[2] == 𝒊[3]
                A[𝒊] = 𝐭[𝒊[1]]
            end
        end

        # TensorDecompositions's dense method
        lbd, x = sshopm(A, 0, tol=tolerance, maxiter=maxIteration)
        x = abs.(x)

        # TensorDecompositions's sparse method
        As = SparseArray(A)
        lbdsparse, y = sshopm(As, 0, tol=tolerance, maxiter=maxIteration)
        y = abs.(y)

        # hopm_canonical
        zero𝐓 = BlockedTensor([ValueBlock(zeros(valN,valN,valN))], [IndexBlock(NTuple{3,Int}[])], size(𝐓))
        energy, Z = hopm_canonical(𝐭, zero𝐓, 𝑻, rand(valN, idxN), tolerance, maxIteration)
        z = reshape(Z, length(Z))

        @test lbd ≈ lbdsparse ≈ energy
        @test vecnorm(y - x) < 5tolerance
        @test vecnorm(z - x) < 5tolerance
        @test vecnorm(z - y) < 5tolerance
    end

    @testset "2nd order mixed" begin
        𝐭 = rand(valN, idxN)
        𝐌 = rand(valN, idxN)
        Ex, X = hopm_canonical(𝐭, 𝐓, 𝐌, tolerance, maxIteration)
        Ey, Y = hopm_mixed(𝐭, 𝐓, 𝐌, :vecnorm, tolerance, maxIteration)
        Ez, Z = hopm_mixed(𝐭, 𝐓, 𝐌, :column, tolerance, maxIteration)
        println("The following values should not deviate too much from each other: ")
        @show Ex, Ey, Ez/idxN
        @show vecnorm(X - Y)
        @show vecnorm(X - Z)
        @show vecnorm(Y - Z)
    end

    @testset "3rd order mixed" begin
        𝐭 = 10*rand(valN, idxN)
        𝐌 = rand(valN, idxN)
        Ex, X = hopm_canonical(𝐭, 𝐓, 𝑻, 𝐌, tolerance, maxIteration)
        Ey, Y = hopm_mixed(𝐭, 𝐓, 𝑻, 𝐌, :vecnorm, tolerance, maxIteration)
        Ez, Z = hopm_mixed(𝐭, 𝐓, 𝑻, 𝐌, :column, tolerance, maxIteration)
        println("The following values should not deviate too much from each other: ")
        @show Ex, Ey, Ez/idxN
        @show vecnorm(X - Y)
        @show vecnorm(X - Z)
        @show vecnorm(Y - Z)
    end
end
# end
