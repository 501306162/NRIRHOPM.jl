using TensorOperations

@testset "tensors" begin
    @testset "ValueBlock" begin
        v = rand(3,3)
        x = ValueBlock(v)
        @test size(x) == size(v)
        @test x[3] == v[3]
        @test x[1:3] == v[1:3]
        @test x[2,2] == v[2,2]
        @test x == ValueBlock(v)
    end

    @testset "ValueBlock" begin
        x = IndexBlock([(1,2),(2,2),(3,1)])
        @test size(x) == (3,)
        @test x[2] == x.idxs[2]
        @test_throws BoundsError x[2,2]
    end

    @testset "BlockedTensor" begin
        v = rand(3,3)
        vals = [ValueBlock(v)]
        idxs = [IndexBlock([(1,2),(2,2),(3,1)])]
        dims = (3,3,3,2)
        x = BlockedTensor(vals,idxs,dims)
        @test x[:,1,:,2] == x[:,2,:,2] == x[:,3,:,1] == v
        @test x == BlockedTensor(vals,idxs,dims)
    end

    valN = 3
    idxN = 4
    @testset "4th order contract" begin
        # construct a BlockedTensor
        vals = [ValueBlock(rand(valN, valN))]
        index = NTuple{2,Int}[]
        for 𝒊 in CartesianRange((idxN,idxN))
            if 𝒊[1] < 𝒊[2]
                push!(index, 𝒊.I)
            end
        end
        idxs = [IndexBlock(index)]
        dims = (valN, idxN, valN, idxN)
        x = BlockedTensor(vals, idxs, dims)
        # convert x to a full symmetric tensor
        y = full(x)
        @test y[1,2,2,1] == y[2,1,1,2] == x[2,1,1,2]
        @test y[1,3,2,1] == y[2,1,1,3] == x[2,1,1,3]
        @test y[3,3,1,2] == y[1,2,3,3] == x[1,2,3,3]
        # test `contract`
        r = rand(valN, idxN)
        @tensor v[a,i] := y[a,i,b,j] * r[b,j]
        @test x ⊙ r ≈ v
    end

    @testset "6th order contract" begin
        # construct a BlockedTensor
        vals = [ValueBlock(rand(valN, valN, valN))]
        index = NTuple{3,Int}[]
        for 𝒊 in CartesianRange((idxN,idxN,idxN))
            if 𝒊[1] < 𝒊[2] < 𝒊[3]
                push!(index, 𝒊.I)
            end
        end
        idxs = [IndexBlock(index)]
        dims = (valN, idxN, valN, idxN, valN, idxN)
        x = BlockedTensor(vals, idxs, dims)
        # convert x to a full symmetric tensor
        y = full(x)
        @test y[1,2,2,3,2,1] == y[2,3,2,1,1,2] == y[2,1,1,2,2,3] == x[2,1,1,2,2,3]
        @test y[1,3,2,1,2,4] == y[2,4,1,3,2,1] == y[2,1,1,3,2,4] == x[2,1,1,3,2,4]
        @test y[3,3,1,2,1,4] == y[3,3,1,4,1,2] == y[1,2,3,3,1,4] == x[1,2,3,3,1,4]
        # test `contract`
        r = rand(valN, idxN)
        @tensor v[a,i] := y[a,i,b,j,c,k] * r[b,j] * r[c,k]
        @test x ⊙ r ≈ v
    end

    @testset "8th order contract" begin
        # construct a BlockedTensor
        vals = [ValueBlock(rand(valN, valN, valN, valN))]
        index = NTuple{4,Int}[]
        for 𝒊 in CartesianRange((idxN,idxN,idxN,idxN))
            if 𝒊[1] < 𝒊[2] < 𝒊[3] < 𝒊[4]
                push!(index, 𝒊.I)
            end
        end
        idxs = [IndexBlock(index)]
        dims = (valN, idxN, valN, idxN, valN, idxN, valN, idxN)
        x = BlockedTensor(vals, idxs, dims)
        # convert x to a full symmetric tensor
        y = full(x)
        @test y[1,2,2,3,3,4,2,1] == y[3,4,2,3,2,1,1,2] == y[2,1,1,2,2,3,3,4] == x[2,1,1,2,2,3,3,4]
        @test y[3,4,1,2,2,3,2,1] == y[2,3,3,4,2,1,1,2] == y[2,1,1,2,2,3,3,4] == x[2,1,1,2,2,3,3,4]
        @test y[2,3,1,2,3,4,2,1] == y[1,2,2,1,2,3,3,4] == y[2,1,1,2,2,3,3,4] == x[2,1,1,2,2,3,3,4]
        # test `contract`
        r = rand(valN, idxN)
        @tensor v[a,i] := y[a,i,b,j,c,k,d,m] * r[b,j] * r[c,k] * r[d,m]
        @test x ⊙ r ≈ v
    end
end
