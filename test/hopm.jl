using TensorDecompositions

@testset "hopm" begin
    @testset "symmetric matrix(pure 2nd order)" begin
        n = 100
        A = Symmetric(rand(n,n))
        A -= diagm(diag(A))

        # build-in eigenpair function
        d, v, nconv, niter, nmult, resid = eigs(A, nev=1, which=:LM)
        v = abs.(v[:])

        # TensorDecompositions's dense method
        lbd, x = sshopm(A, 0)
        x = abs.(x)

        # TensorDecompositions's sparse method
        As = SparseArray(A)
        lbdsparse, y = sshopm(As, 0)
        y = abs.(y)

        # SSTensor
        data = Float64[]
        index = NTuple{2,Int}[]
        for i = 1:n, j = i+1:n
            push!(data, A[i,j])
            push!(index, (i,j))
        end
        score, z = hopm(zeros(n), SSTensor(data,index,(n,n)), rand(n))

        @test d[] ≈ lbd ≈ lbdsparse ≈ score
        @test vecnorm(z - x) < 1e-5
        @test vecnorm(z - y) < 1e-5
        @test vecnorm(z - v) < 1e-5
    end

    @testset "symmetric tensor(pure 3rd order)" begin
        n = 100
        a = rand(n)
        A = kron(a, a', a)
        A = reshape(A, n, n, n)

        data = Float64[]
        index = NTuple{3,Int}[]
        for i = 1:n, j = 1:n, k = 1:n
            if i == j || i == k || j == k
                A[i,j,k] = 0
            elseif i < j < k
                push!(data, A[i,j,k])
                push!(index, (i,j,k))
            end
        end

        # TensorDecompositions's dense method
        lbd, x = sshopm(A, 0)
        x = abs.(x)

        # TensorDecompositions's sparse method
        As = SparseArray(A)
        lbdsparse, y = sshopm(As, 0)
        y = abs.(y)

        # SSTensor
        score, z = hopm(zeros(n), SSTensor([0.0],[(1,1)],(n,n)), SSTensor(data,index,(n,n,n)), rand(n))

        @test lbd ≈ lbdsparse ≈ score
        @test vecnorm(z - x) < 1e-5
        @test vecnorm(z - y) < 1e-5
    end

    @testset "symmetric matrix(non-zero datacost)" begin
        n = 100
        A = Symmetric(rand(n,n))
        A = convert(Array, A)
        # A -= diagm(diag(A))
        #
        # datacost = rand(n)
        # A += diagm(datacost)

        # build-in eigenpair function
        d, v, nconv, niter, nmult, resid = eigs(A, nev=1, which=:LM)
        v = abs.(v[:])

        # TensorDecompositions's dense method
        lbd, x = sshopm(A, 0)
        x = abs.(x)

        # TensorDecompositions's sparse method
        As = SparseArray(A)
        lbdsparse, y = sshopm(As, 0)
        y = abs.(y)

        # SSTensor
        data = Float64[]
        index = NTuple{2,Int}[]
        for i = 1:n, j = i+1:n
            push!(data, A[i,j])
            push!(index, (i,j))
        end
        score, z = hopm(diag(A), SSTensor(data,index,(n,n)), rand(n))

        @test d[] ≈ lbd ≈ lbdsparse ≈ score
        @test vecnorm(z - x) < 1e-5
        @test vecnorm(z - y) < 1e-5
        @test vecnorm(z - v) < 1e-5
    end

    @testset "symmetric tensor(non-zero datacost)" begin
        # n = 100
        # a = rand(n)
        # A = kron(a, a', a)
        # A = reshape(A, n, n, n)
        #
        # data = Float64[]
        # index = NTuple{3,Int}[]
        # for i = 1:n, j = 1:n, k = 1:n
        #     if i == j || i == k || j == k
        #         A[i,j,k] = 0
        #     elseif i < j < k
        #         push!(data, A[i,j,k])
        #         push!(index, (i,j,k))
        #     end
        # end
        #
        # # TensorDecompositions's dense method
        # lbd, x = sshopm(A, 0)
        # x = abs.(x)
        #
        # # TensorDecompositions's sparse method
        # As = SparseArray(A)
        # lbdsparse, y = sshopm(As, 0)
        # y = abs.(y)
        #
        # # SSTensor
        # score, z = hopm(zeros(n), SSTensor([0.0],[(1,1)],(n,n)), SSTensor(data,index,(n,n,n)), rand(n))
        #
        # @test lbd ≈ lbdsparse ≈ score
        # @test vecnorm(z - x) < 1e-5
        # @test vecnorm(z - y) < 1e-5
    end
end
