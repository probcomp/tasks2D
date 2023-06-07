### This is all copied from Mirko's SLAM tutorial. ###

function line_intersection(x1, x2, x1′, x2′, y1, y2, y1′, y2′)
    dx1, dx2 = x1′ - x1, x2′ - x2
    dy1, dy2 = y1′ - y1, y2′ - y2
    
    v1 = (x1 - y1)
    v2 = (x2 - y2)
    
    a, b = -dx1, dy1
    c, d = -dx2, dy2
    
    det = a*d - c*b
    
    if det == 0
        return Inf,Inf
    end
    
    s = 1/det*(  d*v1 - b*v2)
    t = 1/det*(- c*v1 + a*v2)

    return s,t
end;
function cast_cpu!(Z, X, P, fov=2π)
    num_a = size(Z,2)
    r     = fov/(num_a-1)

    for i = 1:size(X,1), j = 1:size(P,1)
        # convert everything into pose coords
        x1 , x2  = X[i,1] - P[j,1], X[i,2] - P[j,2]
        x1′, x2′ = X[i,3] - P[j,1], X[i,4] - P[j,2]
        dx1, dx2 = x1′-x1, x2′-x2
        y1 , y2  = 0, 0
        a1 = atan(x2 , x1 ) - P[j,3]
        a2 = atan(x2′, x1′) - P[j,3]
        a1 = mod(a1 + π, 2π) - π
        a2 = mod(a2 + π, 2π) - π

        # Ensure a1 < a2
        if a1 > a2
            a1, a2 = a2, a1
        end
        
        # Check if we cross from `-π+a` to `π-b`
        if a2 - a1 > π
            # Get the start end end bin
            zero = - fov/2;
            k1 = Int(floor((-π + r/2 - zero)/r))+1
            k2 = Int(floor((a1 + r/2 - zero)/r))+1
            
            k1′ = Int(floor((a2 + r/2 - zero)/r))+1
            k2′ = Int(floor((π + r/2 - zero)/r))+1
            
            ks = ((k1,k2),(k1′,k2′))
        else
            # Get the start end end bin
            zero = - fov/2;
            k1 = Int(floor((a1 + r/2 - zero)/r))+1
            k2 = Int(floor((a2 + r/2 - zero)/r))+1
            
            ks = ((k1,k2),)
            
        end
        
        
        for (k1,k2) in ks, k = k1:k2
            if !(1 <= k <= num_a)
               continue 
            end

            a = zero + (k-1)*r + P[j,3] 
            y1′, y2′ = cos(a), sin(a)

            s, t = line_intersection(x1, x2, x1′, x2′, y1, y2, y1′, y2′)
            if 0 < t && 0 <= s <= 1    
                @inbounds Z[j,k] = min(Z[j,k], t)
            end
        end
    end
    return
end


"""
```julia
    zs = cast_cpu(ps, segs; fov=2π, num_a::Int=361, zmax::Float64=Inf)
```
Computes depth measurements `zs` with respect to a family of stacked poses `ps`
and family of stacked line segments `segs_` along a fixed number `num_a` of
equidistantly spaced angles in the field of view `fov`.

Arguments:
 - `ps`: Stacked poses `(k, 3)`
   > Each pose is a `[x, y, θ]` vector.
 - `segs`: Stacked line segments `(n, 4)`
 - ...

Returns:
 - `zs`: Depth measurements in the field of view `(k, num_a)`
"""
function cast_cpu(ps, segs; fov=2π, num_a::Int=361, zmax::Float64=Inf)
    zs = fill(zmax, size(ps, 1), num_a)
    cast_cpu!(zs, segs, ps, fov)
    return zs
end;