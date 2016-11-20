using NRIRHOPM, TensorOperations
using Base.Test

import NRIRHOPM: pcontract

# test for second-order PSSTensor
info("Testing second-order PSSTensor:")
n = 100
A = Symmetric(rand(n,n))
A -= diagm(diag(A))
x = rand(n)

data = Float64[]
index = NTuple{2,Int}[]
for i = 1:n, j = i+1:n
    push!(data, A[i,j])
    push!(index, (i,j))
end

@test vecnorm(pcontract(PSSTensor(data, index, size(A)), x) - tensorcontract(A, [1,2], x, [2])) < 1e-5
@test vecnorm(PSSTensor(data, index, size(A)) ⊙ x - tensorcontract(A, [1,2], x, [2])) < 1e-5
println("Passed.")

# test for third-order PSSTensor
info("Testing third-order PSSTensor:")
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

x = rand(n)

PSSTensor(data, index, size(A)).data

v = tensorcontract(A, [1,2,3], x, [3])

@test vecnorm(pcontract(PSSTensor(data, index, size(A)), x) - tensorcontract(v, [1,2], x, [2])) < 1e-5
@test vecnorm(PSSTensor(data, index, size(A)) ⊙ x - tensorcontract(v, [1,2], x, [2])) < 1e-5
println("Passed.")
