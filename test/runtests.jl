using NRIRHOPM
using Base.Test

img = Float64[ 1  2  3  4  5;
              10  9  8  7  6;
              11 12 13 14 15;
              16 17 18 19 20;
              21 22 23 24 25]

movimg = Float64[ 1  2  3  4  5;
                 10  9  8 12  6;
                 11  7 13 18 15;
                 16 17 14 19 20;
                 21 22 23 24 25]

deformableWindow = [(i,j) for i in -2:2, j in -2:2]

# algorithm
@time E, y = integerhopm(img, movimg, [deformableWindow...])

yMat = reshape(y, length(img), length(deformableWindow))

deformed = Array{Tuple}(size(img))

for i in 1:length(img)
    a = ind2sub(size(img),i)
    b = deformableWindow[findmax(yMat[i,:])[2]]
    deformed[a...] = b
end

@test deformed == [(0,0)  (0,0)  (0,0)  (0,0) (0,0);
                   (0,0)  (0,0)  (0,0) (1,-2) (0,0);
                   (0,0) (-1,2)  (0,0) (1,-1) (0,0);
                   (0,0)  (0,0) (-1,1)  (0,0) (0,0);
                   (0,0)  (0,0)  (0,0)  (0,0) (0,0)]
