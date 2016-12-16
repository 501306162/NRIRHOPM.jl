info("Testing 8-connected neighborhood: ")
# test for simplest 8-connected neighborhood
# pairwise cliques
#  1  4  7       11  12  13
#  2  5  8  <=>  21  22  23
#  3  6  9       31  32  33
index = neighbors(Connected8{2}, (3,3))

pixel11 = []
pixel21 = [(2,1)]
pixel31 = [(3,2)]
pixel12 = [(4,1), (4,2)]
pixel22 = [(5,1), (5,2), (5,3), (5,4)]
pixel32 = [(6,2), (6,3), (6,5)]
pixel13 = [(7,4), (7,5)]
pixel23 = [(8,4), (8,5), (8,6), (8,7)]
pixel33 = [(9,5), (9,6), (9,8)]

@test index == [pixel11; pixel21; pixel31; pixel12; pixel22; pixel32; pixel13; pixel23; pixel33]

# trey cliques
#  1  4  7       11  12  13
#  2  5  8  <=>  21  22  23
#  3  6  9       31  32  33
◸, ◹, ◺, ◿ = neighbors(Connected8{3}, (3,3))

@test ◸ == [(1,4,2), (2,5,3), (4,7,5), (5,8,6)]
@test ◹ == [(4,1,5), (5,2,6), (7,4,8), (8,5,9)]
@test ◺ == [(2,5,1), (3,6,2), (5,8,4), (6,9,5)]
@test ◿ == [(5,2,4), (6,3,5), (8,5,7), (9,6,8)]

println("Passed.")
