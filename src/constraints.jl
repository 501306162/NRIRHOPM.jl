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
                          deformableWindow::Matrix{Vector{Int}};
                          metric::AbstractDataCost=SAD(),
                          regularization::AbstractRegularization=TAD(),
                          λ::Real=1,
                          γ::Real=1,
                          τ::Real=Inf,
                          β::Int=0
                         )
    imageLen = length(fixedImg)
    deformLen = length(deformableWindow)

    # tic-tocing
    @time tensor₁ = unaryclique(fixedImg, movingImg, deformableWindow; metric=metric, β=β)
	@time tensor₂ = pairwiseclique(fixedImg, movingImg, deformableWindow; regularization=regularization, γ=γ, τ=τ)
    @time tensor₂ = SharedSparseTensor(share(tensor₂.vals), share(tensor₂.pos), tensor₂.dims)
	@time e, x = hopm(tensor₁, tensor₂, Float64(λ))

    # greedy integer programming?
	y = integerlize(x, imageLen, deformLen)
	S̄ = dot(y, tensor₁ + λ*A_mul_B(tensor₂, share(y)))
	S = Inf
	while S > S̄
		x = integerlize(tensor₁ + λ*A_mul_B(tensor₂, share(y)), imageLen, deformLen)
		S = dot(x, tensor₁ + λ*A_mul_B(tensor₂, share(x)))
		if S > S̄
            @show "S > S̄"
			S̄ = S
			y = x
		end
		@show S, S̄
	end
    return S̄, y
end
