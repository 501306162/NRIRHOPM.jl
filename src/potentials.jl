# potential functions

# SAD: Sum of Absolute Differences
sad{T<:Number}(x::T, y::T) = abs(x-y)




# distance
distance(x::Tuple{Int,Int}, y::Tuple{Int,Int}) = abs(sqrt(x[1]^2 + x[2]^2) - sqrt(y[1]^2 + y[2]^2))




# topology preserving
function topology(s1::Tuple{Int64,Int64}, s2::Tuple{Int64,Int64}, s3::Tuple{Int64,Int64}, a::Tuple{Int64,Int64}, b::Tuple{Int64,Int64}, c::Tuple{Int64,Int64})
    ks1 = (s1[1]+a[1], s1[2]+a[2])
    ks2 = (s2[1]+b[1], s2[2]+b[2])
    ks3 = (s3[1]+c[1], s3[2]+c[2])
    dφ1 = ks2[1] - ks1[1]
    dr1 = s2[1] - s1[1]
    dφ2 = ks2[2] - ks3[2]
    dr2 = s2[2] - s3[2]
    dφ3 = ks2[2] - ks1[2]
    # dr3 = dr1
    dφ4 = ks2[1] - ks3[1]
    # dr4 = dr2
    v = (dφ1/dr1 * dφ2/dr2) - (dφ3/dr1 * dφ4/dr2)
    return v::Float64
end
