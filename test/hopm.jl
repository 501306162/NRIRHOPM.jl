using TensorDecompositions

@testset "hopm" begin
    TOL = rand([1e-5,1e-6,1e-7])
    n = 100
    @testset "symmetric matrix" begin
        A = Symmetric(rand(n,n))
        A = convert(Array, A)

        # build-in eigenpair function
        d, v, nconv, niter, nmult, resid = eigs(A, nev=1, which=:LM, tol=TOL)
        v = abs.(v[:])

        # TensorDecompositions's dense method
        lbd, x = sshopm(A, 0, tol=TOL, maxiter=300)
        x = abs.(x)

        # TensorDecompositions's sparse method
        As = SparseArray(A)
        lbdsparse, y = sshopm(As, 0, tol=TOL, maxiter=300)
        y = abs.(y)

        # SSTensor
        data = Float64[]
        index = NTuple{2,Int}[]
        for i = 1:n, j = i+1:n
            push!(data, A[i,j])
            push!(index, (i,j))
        end
        score, z = hopm(diag(A), SSTensor(data,index,(n,n)), rand(n), TOL)

        @test d[] ≈ lbd ≈ lbdsparse ≈ score
        @test vecnorm(z - x) < TOL
        @test vecnorm(z - y) < TOL
        @test vecnorm(z - v) < TOL
    end

    @testset "symmetric tensor" begin
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
        lbd, x = sshopm(A, 0, tol=TOL, maxiter=300)
        x = abs.(x)

        # TensorDecompositions's sparse method
        As = SparseArray(A)
        lbdsparse, y = sshopm(As, 0, tol=TOL, maxiter=300)
        y = abs.(y)

        # SSTensor
        score, z = hopm(diagA, SSTensor([0.0],[(1,1)],(n,n)), SSTensor(data,index,(n,n,n)), rand(n), TOL)

        @test lbd ≈ lbdsparse ≈ score
        @test vecnorm(z - x) < TOL
        @test vecnorm(z - y) < TOL
    end

    # @testset "constrain row" begin
    #     imageDims = (16,16)
    #     labels = [[(i,j) for i in -2:2, j in -2:2]...]
    #     pixelNum = prod(imageDims)
    #     labelNum = length(labels)
    #     x = rand(pixelNum*labelNum)
    #     X = reshape(x, pixelNum, labelNum)
    #
    #     ss2 = pairwiseclique4validation(imageDims, labels);
    #     bss2 = pairwiseclique(imageDims, labels, TAD());
    #
    #     ss3 = treyclique4validation(imageDims, [[[i,j] for i in -2:2, j in -2:2]...]);
    #     bss3 = treyclique(imageDims, labels, TP());
    #
    #     ssScore, v = hopm(zeros(pixelNum*labelNum), bss2, bss3, x, TOL)
    #     bssScore, V = hopm(zeros(pixelNum*labelNum), bss2, bss3, X, TOL)
    #
    #     @test ssScore ≈ bssScore
    #     @test vecnorm(v - reshape(V, pixelNum*labelNum)) < TOL
    # end
end
