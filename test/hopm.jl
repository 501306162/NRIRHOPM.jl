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

    #
    # @testset "2nd order mixed" begin
    #     A = Symmetric(rand(n,n))
    #     A = convert(Array, A)
    #     # SSTensor
    #     data = Float64[]
    #     index = NTuple{2,Int}[]
    #     for i = 1:n, j = i+1:n
    #         push!(data, A[i,j])
    #         push!(index, (i,j))
    #     end
    #     v = rand(n)
    #     score, x = hopm_canonical(diag(A), SSTensor(data,index,(n,n)), v, tolerance, maxIteration)
    #     energy, y = hopm_mixed(diag(A), SSTensor(data,index,(n,n)), v, tolerance, maxIteration)
    #     @test vecnorm(x - y) < 0.1    # this is expected with n=100 since these two algorithms are not exactly the same.
    # end
    #
    # @testset "3rd order mixed" begin
    #     a = rand(n)
    #     A = kron(a, a', a)
    #     A = reshape(A, n, n, n)
    #
    #     diagA = zeros(n)
    #     data = Float64[]
    #     index = NTuple{3,Int}[]
    #     for i = 1:n, j = 1:n, k = 1:n
    #         if i == j == k
    #             diagA[i] = A[i,j,k]
    #         elseif i == j || i == k || j == k
    #             A[i,j,k] = 0
    #         elseif i < j < k
    #             push!(data, A[i,j,k])
    #             push!(index, (i,j,k))
    #         end
    #     end
    #     v = rand(n)
    #     score, x = hopm_canonical(diagA, SSTensor([0.0],[(1,1)],(n,n)), SSTensor(data,index,(n,n,n)), v, tolerance, maxIteration)
    #     energy, y = hopm_mixed(diagA, SSTensor([0.0],[(1,1)],(n,n)), SSTensor(data,index,(n,n,n)), v, tolerance, maxIteration)
    #     @test vecnorm(x - y) < 0.01    # this is expected with n=100 since these two algorithms are not exactly the same.
    # end
    #
    # @testset "2nd order constrain vecnorm" begin
    #     imageDims = (5,5)
    #     labels = [[(i,j) for i in -1:1, j in -1:1]...]
    #     pixelNum = prod(imageDims)
    #     labelNum = length(labels)
    #     x = rand(pixelNum*labelNum)
    #     X = reshape(x, pixelNum, labelNum)
    #
    #     ss = pairwiseclique4validation(imageDims, labels);
    #     bss = pairwiseclique(imageDims, labels, TAD());
    #
    #     h = 10*rand(pixelNum*labelNum)
    #     ssScore, v = hopm_mixed(h, ss, x, tolerance, maxIteration)
    #     bssScore, V = hopm_mixed(h, bss, X, tolerance, maxIteration, false)
    #
    #     @test ssScore ≈ bssScore
    #     @test vecnorm(v - reshape(V, pixelNum*labelNum)) < tolerance
    # end
    #
    # @testset "2nd order constrain row" begin
    #     imageDims = (5,5)
    #     labels = [[(i,j) for i in -1:1, j in -1:1]...]
    #     pixelNum = prod(imageDims)
    #     labelNum = length(labels)
    #     x = rand(pixelNum*labelNum)
    #     X = reshape(x, pixelNum, labelNum)
    #
    #     ss = pairwiseclique4validation(imageDims, labels);
    #     bss = pairwiseclique(imageDims, labels, TAD());
    #
    #     h = 10*rand(pixelNum*labelNum)
    #     vecnormScore, v = hopm_mixed(h, ss, x, tolerance, maxIteration)
    #     constrainRowScore, R = hopm_mixed(h, bss, X, tolerance, maxIteration, true)
    #
    #     @show vecnormScore, constrainRowScore
    #     @show vecnorm(v - reshape(R, pixelNum*labelNum))
    # end
    #
    # @testset "3rd order constrain vecnorm" begin
    #     imageDims = (3,3)
    #     labels = [[(i,j) for i in -1:1, j in -1:1]...]
    #     pixelNum = prod(imageDims)
    #     labelNum = length(labels)
    #     x = rand(pixelNum*labelNum)
    #     X = reshape(x, pixelNum, labelNum)
    #
    #     ss2 = pairwiseclique4validation(imageDims, labels);
    #     bss2 = pairwiseclique(imageDims, labels, TAD());
    #
    #     ss3 = treyclique4validation(imageDims, [[[i,j] for i in -1:1, j in -1:1]...]);
    #     bss3 = treyclique(imageDims, labels, TP2D());
    #
    #     h = 10*rand(pixelNum*labelNum)
    #     ssScore, v = hopm_mixed(h, ss2, ss3, x, tolerance, maxIteration)
    #     bssScore, V = hopm_mixed(h, bss2, bss3, X, tolerance, maxIteration, false)
    #
    #     @test ssScore ≈ bssScore
    #     @test vecnorm(v - reshape(V, pixelNum*labelNum)) < tolerance
    # end
    #
    # @testset "3rd order constrain row" begin
    #     imageDims = (3,3)
    #     labels = [[(i,j) for i in -1:1, j in -1:1]...]
    #     pixelNum = prod(imageDims)
    #     labelNum = length(labels)
    #     x = rand(pixelNum*labelNum)
    #     X = reshape(x, pixelNum, labelNum)
    #
    #     ss2 = pairwiseclique4validation(imageDims, labels);
    #     bss2 = pairwiseclique(imageDims, labels, TAD());
    #
    #     ss3 = treyclique4validation(imageDims, [[[i,j] for i in -1:1, j in -1:1]...]);
    #     bss3 = treyclique(imageDims, labels, TP2D());
    #
    #     h = 10*rand(pixelNum*labelNum)
    #     vecnormScore, v = hopm_mixed(h, ss2, ss3, x, tolerance, maxIteration)
    #     constrainRowScore, R = hopm_mixed(h, bss2, bss3, X, tolerance, maxIteration, true)
    #
    #     @show vecnormScore, constrainRowScore
    #     @show vecnorm(v - reshape(R, pixelNum*labelNum))
    # end
end
# end
