# functions for enforcing integer constrains

# enforce integer constrains
function integerlize(xVec::Vector, imageLen::Int, deformLen::Int)
    xMat = reshape(xVec, imageLen, deformLen)
    yMat = zeros(imageLen,deformLen)
    for i in 1:imageLen
        yMat[i, findmax(xMat[i,:])[2]] = 1
    end
    return reshape(yMat, imageLen*deformLen)
end

function integerhopm{T,N}(fixedImg::Array{T,N},
                          movingImg::Array{T,N},
                          deformers::Vector{Tuple{Int,Int}};
                          lambda::Float64=0.05,
                          theta::Float64=0.0,
                          delta::Float64=1e2
                         )
    imageLen = length(fixedImg)
    deformLen = length(deformers)
    @time tensor₁ = unaryclique(fixedImg, movingImg, deformers, δ=delta)

	@show "pairwiseclique:"
	@time tensor₂ = pairwiseclique(size(fixedImg), deformers, δ=delta)
    @time tensor₂ = SharedSparseTensor(share(tensor₂.vals), share(tensor₂.pos), tensor₂.dims)

	@time e, x = hopm(tensor₁, tensor₂, lambda)
	y = integerlize(x, imageLen, deformLen)
	S̄ = dot(y, tensor₁ + lambda*A_mul_B(tensor₂, share(y)))
	S = typemax(Float64)
	while S > S̄
		x = integerlize(tensor₁ + lambda*A_mul_B(tensor₂, share(y)), imageLen, deformLen)
		S = dot(x, tensor₁ + lambda*A_mul_B(tensor₂, share(x)))
		if S > S̄
            @show "S > S̄"
			S̄ = S
			y = x
		end
		@show S, S̄
	end
    return S̄, y
end
