# functions for enforcing integer constrains

# enforce integer constrains
function integerlize(xVec::Vector, imageLen::Integer, deformLen::Integer)
    xMat = reshape(xVec, imageLen, deformLen)
    yMat = zeros(imageLen,deformLen)
    for i in 1:imageLen
        yMat[i, findmax(xMat[i,:])[2]] = 1
    end
    return reshape(yMat, imageLen*deformLen)
end

function integerhopm{T,N}(fixedImg::Array{T,N},
                          movingImg::Array{T,N},
                          deformers::Vector{Tuple{Int64,Int64}};
                          lambda::Float64=0.05,
                          theta::Float64=0.0,
                          delta::Float64=1e2
                         )
    imageLen = length(fixedImg)
    deformLen = length(deformers)
    @time tensor₁ = unaryclique(fixedImg, movingImg, deformers, δ=delta)

	if theta == 0
		if lambda == 0
			@time e, x = hopm(tensor₁)
			y = integerlize(x, imageLen, deformLen)
			S̄ = dot(y, A_mul_B(tensor₁, y))
			S = typemax(Float64)
			while S > S̄
				x = integerlize(A_mul_B(tensor₁, y), imageLen, deformLen)
				S = dot(x, A_mul_B(tensor₁, x))
				if S > S̄
					@show "λ=0"
					S̄ = S
					y = x
				end
				@show S, S̄
			end
		else
			@show "pairwiseclique:"
			@time tensor₂ = pairwiseclique(size(fixedImg), deformers, δ=delta)
			@time e, x = hopm(tensor₁, tensor₂, lambda)
			y = integerlize(x, imageLen, deformLen)
			S̄ = dot(y, A_mul_B(tensor₁, y) + lambda*A_mul_B(tensor₂, y))
			S = typemax(Float64)
			while S > S̄
				x = integerlize(A_mul_B(tensor₁, y) + lambda*A_mul_B(tensor₂, y), imageLen, deformLen)
				S = dot(x, A_mul_B(tensor₁, x) + lambda*A_mul_B(tensor₂, x))
				if S > S̄
					@show "λ!=0, θ==0"
					S̄ = S
					y = x
				end
				@show S, S̄
			end
		end
	else
        @show "pairwiseclique:"
        @time tensor₂ = pairwiseclique(size(fixedImg), deformers, δ=delta)
        @show "treyclique:"
        @time tensor₃ = treyclique(size(fixedImg), deformers)
        @time e, x = hopm(tensor₁, tensor₂, tensor₃, lambda, theta)
        y = integerlize(x, imageLen, deformLen)
        S̄ = dot(y, A_mul_B(tensor₁, y) + lambda*A_mul_B(tensor₂, y) + theta*A_mul_B(tensor₃, y))
        S = typemax(Float64)
        while S > S̄
            x = integerlize(A_mul_B(tensor₁, y) + lambda*A_mul_B(tensor₂, y) + theta*A_mul_B(tensor₃, y), imageLen, deformLen)
            S = dot(x, A_mul_B(tensor₁, x) + lambda*A_mul_B(tensor₂, x) + theta*A_mul_B(tensor₃, y))
            if S > S̄
                @show "λ!=0"
                S̄ = S
                y = x
            end
            @show S, S̄
        end
    end
    return S̄, y
end
