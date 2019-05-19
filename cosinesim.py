
import math

v1 = {"1":4,"2":5,"5":1}
v2 = {"2":3,"5":1}
def cosine_similarity(v1,v2):

    allDim = set(list(v1.keys()) + list(v2.keys()))
    dotproduct = 0
    euclidianDistanceV1 =0
    euclidianDistanceV2 =0

    for dim in v1:
        rating = v1[dim]
        euclidianDistanceV1 += (rating * rating)
    euclidianDistanceV1 =math.sqrt(euclidianDistanceV1)

    for dim in v2:
        rating = v2[dim]
        euclidianDistanceV2 += (rating * rating)
    euclidianDistanceV2 =math.sqrt(euclidianDistanceV2)

    for dim in allDim:
        if dim in v1:
            a = v1[dim]
        else:
            a = 0

        if dim in v2:
            b = v2[dim]
        else:
            b = 0

        dotproduct += (a * b)

    return (dotproduct/(euclidianDistanceV1*euclidianDistanceV2))

v1 = {"1":4,"2":5}
v2 = {"2":3,"5":1}
s =cosine_similarity(v1,v2)
print( s )
