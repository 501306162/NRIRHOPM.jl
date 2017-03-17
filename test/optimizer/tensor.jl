using TensorOperations

@testset "tensor" begin
    @testset "ValueBlock" begin
        V = rand(3,3)
        X = ValueBlock(V)
        @test size(X) == size(V)
        @test X[3] == V[3]
        @test X[1:3] == V[1:3]
        @test X[2,2] == V[2,2]
        @test X == ValueBlock(V)
    end

    @testset "ValueBlock" begin
        X = IndexBlock([(1,2),(2,2),(3,1)])
        @test size(X) == (3,)
        @test X[2] == X.idxs[2]
        @test_throws BoundsError X[2,2]
    end

    @testset "CompositeBlockedTensor" begin
        V = rand(3,3)
        vals = [ValueBlock(V)]
        idxs = [IndexBlock([(1,2),(2,2),(3,1)])]
        dims = (3,3,3,2)
        X = CompositeBlockedTensor(vals,idxs,dims)
        @test X[:,1,:,2] == X[:,2,:,2] == X[:,3,:,1] == V
        @test X == CompositeBlockedTensor(vals,idxs,dims)
    end

    valN = 3
    idxN = 4
    @testset "4th order contract" begin
        # construct a CompositeBlockedTensor
        vals = [ValueBlock(rand(valN, valN))]
        index = NTuple{2,Int}[]
        for 𝒊 in CartesianRange((idxN,idxN))
            if 𝒊[1] < 𝒊[2]
                push!(index, 𝒊.I)
            end
        end
        idxs = [IndexBlock(index)]
        dims = (valN, idxN, valN, idxN)
        X = CompositeBlockedTensor(vals, idxs, dims)
        # convert x to a full symmetric tensor
        Y = full(X)
        @test Y[1,2,2,1] == Y[2,1,1,2] == X[2,1,1,2]
        @test Y[1,3,2,1] == Y[2,1,1,3] == X[2,1,1,3]
        @test Y[3,3,1,2] == Y[1,2,3,3] == X[1,2,3,3]
        # test `contract`
        R = rand(valN, idxN)
        @tensor V[a,i] := Y[a,i,b,j] * R[b,j]
        @test X ⊙ R ≈ V

        r = rand(valN*idxN)
        y = reshape(Y, valN*idxN, valN*idxN)
        @tensor v[ai] := y[ai,bj] * r[bj]
        @test X ⊙ r ≈ v
    end

    @testset "6th order contract" begin
        # construct a CompositeBlockedTensor
        vals = [ValueBlock(rand(valN, valN, valN))]
        index = NTuple{3,Int}[]
        for 𝒊 in CartesianRange((idxN,idxN,idxN))
            if 𝒊[1] < 𝒊[2] < 𝒊[3]
                push!(index, 𝒊.I)
            end
        end
        idxs = [IndexBlock(index)]
        dims = (valN, idxN, valN, idxN, valN, idxN)
        X = CompositeBlockedTensor(vals, idxs, dims)
        # convert x to a full symmetric tensor
        Y = full(X)
        @test Y[1,2,2,3,2,1] == Y[2,3,2,1,1,2] == Y[2,1,1,2,2,3] == X[2,1,1,2,2,3]
        @test Y[1,3,2,1,2,4] == Y[2,4,1,3,2,1] == Y[2,1,1,3,2,4] == X[2,1,1,3,2,4]
        @test Y[3,3,1,2,1,4] == Y[3,3,1,4,1,2] == Y[1,2,3,3,1,4] == X[1,2,3,3,1,4]
        # test `contract`
        R = rand(valN, idxN)
        @tensor V[a,i] := Y[a,i,b,j,c,k] * R[b,j] * R[c,k]
        @test X ⊙ R ≈ V

        r = rand(valN*idxN)
        y = reshape(Y, valN*idxN, valN*idxN, valN*idxN)
        @tensor v[ai] := y[ai,bj,ck] * r[bj] * r[ck]
        @test X ⊙ r ≈ v
    end

    @testset "8th order contract" begin
        # construct a CompositeBlockedTensor
        vals = [ValueBlock(rand(valN, valN, valN, valN))]
        index = NTuple{4,Int}[]
        for 𝒊 in CartesianRange((idxN,idxN,idxN,idxN))
            if 𝒊[1] < 𝒊[2] < 𝒊[3] < 𝒊[4]
                push!(index, 𝒊.I)
            end
        end
        idxs = [IndexBlock(index)]
        dims = (valN, idxN, valN, idxN, valN, idxN, valN, idxN)
        X = CompositeBlockedTensor(vals, idxs, dims)
        # convert x to a full symmetric tensor
        Y = full(X)
        @test Y[1,2,2,3,3,4,2,1] == Y[3,4,2,3,2,1,1,2] == Y[2,1,1,2,2,3,3,4] == X[2,1,1,2,2,3,3,4]
        @test Y[3,4,1,2,2,3,2,1] == Y[2,3,3,4,2,1,1,2] == Y[2,1,1,2,2,3,3,4] == X[2,1,1,2,2,3,3,4]
        @test Y[2,3,1,2,3,4,2,1] == Y[1,2,2,1,2,3,3,4] == Y[2,1,1,2,2,3,3,4] == X[2,1,1,2,2,3,3,4]
        # test `contract`
        R = rand(valN, idxN)
        @tensor V[a,i] := Y[a,i,b,j,c,k,d,m] * R[b,j] * R[c,k] * R[d,m]
        @test X ⊙ R ≈ V

        r = rand(valN*idxN)
        y = reshape(Y, valN*idxN, valN*idxN, valN*idxN, valN*idxN)
        @tensor v[ai] := y[ai,bj,ck,dm] * r[bj] * r[ck] * r[dm]
        @test X ⊙ r ≈ v
    end
end
