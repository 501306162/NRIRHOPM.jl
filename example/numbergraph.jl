addprocs(2)
using PyPlot
using NRIRHOPM

img = Float64[ 1  2  3  4  5;
              10  9  8  7  6;
              11 12 13 14 15;
              16 17 18 19 20;
              21 22 23 24 25]

movimg = Float64[ 1  2  3  4  5;
                 10  9  8  7  6;
                 11 13 12 14 15;
                 16 17 18 24 20;
                 21 22 23 19 25]

deformableWindow = [(i,j) for i in -2:2, j in -2:2]

# algorithm
@time E, y = integerhopm(img, movimg, [deformableWindow...])

yMat = reshape(y, length(img), length(deformableWindow))

deformgrid = Array{Tuple}(size(img))

for i in 1:length(img)
	a = ind2sub(size(img),i)
	b = deformableWindow[findmax(yMat[i,:])[2]]
	deformgrid[a...] = b
end

@show deformgrid

# meshgrid
x = [i for i in 1:5, j in 1:5]
y = [j for i in 1:5, j in 1:5]

dx = [ deformgrid[i,j][1] for i in 1:5, j in 1:5]
dy = [ deformgrid[i,j][2] for i in 1:5, j in 1:5]

plot_wireframe(x+dx,y+dy,ones(5,5))
