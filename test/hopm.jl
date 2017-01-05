using TensorDecompositions

@testset "hopm" begin
    # hopm(𝐇¹, 𝐇²) -> (s, 𝐯)
    n = 100
    A = Symmetric(rand(n,n))
    A -= diagm(diag(A))
    x = rand(n)

    data𝐇¹ = zeros(n)
    data𝐇² = Float64[]
    index𝐇² = NTuple{2,Int}[]
    for i = 1:n, j = i+1:n
        if i == j
            data𝐇¹[i] = A[i,j]
        else
            push!(data𝐇², A[i,j])
            push!(index𝐇², (i,j))
        end
    end
    lbd, x = sshopm(A, 0)
    score, y = hopm(data𝐇¹, SSTensor(data𝐇², index𝐇², (n,n)))
    # known issue(#5) due to numerical errors
    @show vecnorm(lbd - score)
    @show vecnorm(x - y)

    # hopm(𝐇¹, 𝐇², 𝐇³) -> (s, 𝐯)
    n = 100
    a = rand(n)
    A = kron(a, a', a)
    A = reshape(A, n, n, n)

    data𝐇¹ = zeros(n)
    data𝐇³ = Float64[]
    index𝐇³ = NTuple{3,Int}[]

    for i = 1:n, j = 1:n, k = 1:n
        if i == j && i == k
            data𝐇¹[i] = A[i,j,k]
        elseif i == j || i == k || j == k
            A[i,j,k] = 0
        elseif i < j < k
            push!(data𝐇³, A[i,j,k])
            push!(index𝐇³, (i,j,k))
        end
    end

    lbd, x = sshopm(A, 0)
    score, y = hopm(data𝐇¹, SSTensor([0.0], [(1,1)], (n,n)), SSTensor(data𝐇³, index𝐇³, (n,n,n)))
    # known issue(#5) due to numerical errors
    @show vecnorm(lbd - score)
    @show vecnorm(x - y)
end
