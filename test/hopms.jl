using TensorDecompositions
import NRIRHOPM: hopm_mixed, hopm_canonical

@testset "hopms" begin
    tolerance = 1e-6
    n = 100
    maxIteration = 300
    @testset "2nd order canonical" begin
        A = Symmetric(rand(n,n))
        A = convert(Array, A)

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

        # SSTensor
        data = Float64[]
        index = NTuple{2,Int}[]
        for i = 1:n, j = i+1:n
            push!(data, A[i,j])
            push!(index, (i,j))
        end
        score, z = hopm_canonical(diag(A), SSTensor(data,index,(n,n)), rand(n), tolerance, maxIteration)

        @test d[] ≈ lbd ≈ lbdsparse ≈ score
        @test vecnorm(z - x) < tolerance
        @test vecnorm(z - y) < tolerance
        @test vecnorm(z - v) < tolerance
    end

    @testset "3rd order canonical" begin
        a = rand(n)
        A = kron(a, a', a)
        A = reshape(A, n, n, n)

        diagA = zeros(n)
        data = Float64[]
        index = NTuple{3,Int}[]
        for i = 1:n, j = 1:n, k = 1:n
            if i == j == k
                diagA[i] = A[i,j,k]
            elseif i == j || i == k || j == k
                A[i,j,k] = 0
            elseif i < j < k
                push!(data, A[i,j,k])
                push!(index, (i,j,k))
            end
        end

        # TensorDecompositions's dense method
        lbd, x = sshopm(A, 0, tol=tolerance, maxiter=300)
        x = abs.(x)

        # TensorDecompositions's sparse method
        As = SparseArray(A)
        lbdsparse, y = sshopm(As, 0, tol=tolerance, maxiter=300)
        y = abs.(y)

        # SSTensor
        score, z = hopm_canonical(diagA, SSTensor([0.0],[(1,1)],(n,n)), SSTensor(data,index,(n,n,n)), rand(n), tolerance, maxIteration)

        @test lbd ≈ lbdsparse ≈ score
        @test vecnorm(z - x) < tolerance
        @test vecnorm(z - y) < tolerance
    end

    @testset "2nd order mixed" begin
        A = Symmetric(rand(n,n))
        A = convert(Array, A)
        # SSTensor
        data = Float64[]
        index = NTuple{2,Int}[]
        for i = 1:n, j = i+1:n
            push!(data, A[i,j])
            push!(index, (i,j))
        end
        v = rand(n)
        score, x = hopm_canonical(diag(A), SSTensor(data,index,(n,n)), v, tolerance, maxIteration)
        energy, y = hopm_mixed(diag(A), SSTensor(data,index,(n,n)), v, tolerance, maxIteration)
        @test vecnorm(x - y) < 0.1    # this is expected with n=100 since these two algorithms are not exactly the same.
    end

    @testset "3rd order mixed" begin
        a = rand(n)
        A = kron(a, a', a)
        A = reshape(A, n, n, n)

        diagA = zeros(n)
        data = Float64[]
        index = NTuple{3,Int}[]
        for i = 1:n, j = 1:n, k = 1:n
            if i == j == k
                diagA[i] = A[i,j,k]
            elseif i == j || i == k || j == k
                A[i,j,k] = 0
            elseif i < j < k
                push!(data, A[i,j,k])
                push!(index, (i,j,k))
            end
        end
        v = rand(n)
        score, x = hopm_canonical(diagA, SSTensor([0.0],[(1,1)],(n,n)), SSTensor(data,index,(n,n,n)), v, tolerance, maxIteration)
        energy, y = hopm_mixed(diagA, SSTensor([0.0],[(1,1)],(n,n)), SSTensor(data,index,(n,n,n)), v, tolerance, maxIteration)
        @test vecnorm(x - y) < 0.01    # this is expected with n=100 since these two algorithms are not exactly the same.
    end

    @testset "2nd order constrain vecnorm" begin
        imageDims = (5,5)
        labels = [[(i,j) for i in -1:1, j in -1:1]...]
        pixelNum = prod(imageDims)
        labelNum = length(labels)
        x = rand(pixelNum*labelNum)
        X = reshape(x, pixelNum, labelNum)

        ss = pairwiseclique4validation(imageDims, labels);
        bss = pairwiseclique(imageDims, labels, TAD());

        h = 10*rand(pixelNum*labelNum)
        ssScore, v = hopm_mixed(h, ss, x, tolerance, maxIteration)
        bssScore, V = hopm_mixed(h, bss, X, tolerance, maxIteration, false)

        @test ssScore ≈ bssScore
        @test vecnorm(v - reshape(V, pixelNum*labelNum)) < tolerance
    end

    @testset "2nd order constrain row" begin
        imageDims = (5,5)
        labels = [[(i,j) for i in -1:1, j in -1:1]...]
        pixelNum = prod(imageDims)
        labelNum = length(labels)
        x = rand(pixelNum*labelNum)
        X = reshape(x, pixelNum, labelNum)

        ss = pairwiseclique4validation(imageDims, labels);
        bss = pairwiseclique(imageDims, labels, TAD());

        h = 10*rand(pixelNum*labelNum)
        vecnormScore, v = hopm_mixed(h, ss, x, tolerance, maxIteration)
        constrainRowScore, R = hopm_mixed(h, bss, X, tolerance, maxIteration, true)

        @show vecnormScore, constrainRowScore
        @show vecnorm(v - reshape(R, pixelNum*labelNum))
    end

    @testset "3rd order constrain vecnorm" begin
        imageDims = (3,3)
        labels = [[(i,j) for i in -1:1, j in -1:1]...]
        pixelNum = prod(imageDims)
        labelNum = length(labels)
        x = rand(pixelNum*labelNum)
        X = reshape(x, pixelNum, labelNum)

        ss2 = pairwiseclique4validation(imageDims, labels);
        bss2 = pairwiseclique(imageDims, labels, TAD());

        ss3 = treyclique4validation(imageDims, [[[i,j] for i in -1:1, j in -1:1]...]);
        bss3 = treyclique(imageDims, labels, TP2D());

        h = 10*rand(pixelNum*labelNum)
        ssScore, v = hopm_mixed(h, ss2, ss3, x, tolerance, maxIteration)
        bssScore, V = hopm_mixed(h, bss2, bss3, X, tolerance, maxIteration, false)

        @test ssScore ≈ bssScore
        @test vecnorm(v - reshape(V, pixelNum*labelNum)) < tolerance
    end

    @testset "3rd order constrain row" begin
        imageDims = (3,3)
        labels = [[(i,j) for i in -1:1, j in -1:1]...]
        pixelNum = prod(imageDims)
        labelNum = length(labels)
        x = rand(pixelNum*labelNum)
        X = reshape(x, pixelNum, labelNum)

        ss2 = pairwiseclique4validation(imageDims, labels);
        bss2 = pairwiseclique(imageDims, labels, TAD());

        ss3 = treyclique4validation(imageDims, [[[i,j] for i in -1:1, j in -1:1]...]);
        bss3 = treyclique(imageDims, labels, TP2D());

        h = 10*rand(pixelNum*labelNum)
        vecnormScore, v = hopm_mixed(h, ss2, ss3, x, tolerance, maxIteration)
        constrainRowScore, R = hopm_mixed(h, bss2, bss3, X, tolerance, maxIteration, true)

        @show vecnormScore, constrainRowScore
        @show vecnorm(v - reshape(R, pixelNum*labelNum))
    end
end
