using CairoMakie
using Base.Threads
using LinearAlgebra
using JLD2
using CodecZlib
using Printf
using DelimitedFiles

save_dir = @__DIR__
mkpath(save_dir)

file_path = joinpath(@__DIR__, "test.dat")

N = 200
boxsize = 800
particles = Matrix{Float64}(undef, 4, N)
predictedpositions = Matrix{Float64}(undef, 2, N)
dt = .001

k = 60000000
g = 3000
R = 40
scaling_factor = Int(boxsize / R)
MegaMatrix = zeros(Int, (scaling_factor)^2 + scaling_factor, N)

macro Initialize()
    for i in 1:N
        for j in 1:2
            particles[j, i] = boxsize * rand()
        end
        for j in 3:4
            particles[j, i] = 0.0
        end
    end
end

function positionupdate(id, dx, dy)
    particles[1, id] += dx
    particles[2, id] += dy
end

function velocityupdate(id, dx, dy)
    particles[3, id] += dx
    particles[4, id] += dy
end

function influence(r)
    return max(0, (35 / (16 * R^7)) * ((R)^2 - r^2)^3)
end
function prederivative(r)
    return (35 / (16 * R^7))*(-6*r) * (R-r)^2
    
end
function viscositykernel(r)
    return max(0,45/(pi*R^6) * (R-r))
end

function distance(x, y, id)
    return sqrt((x - predictedpositions[1, id])^2 + (y - predictedpositions[2, id])^2)
end

function updatePredictions()
    Threads.@threads for i in 1:N
        predictedpositions[1,i] = particles[1,i] + particles[3,i] *dt
        predictedpositions[2,i] = particles[2,i] + particles[4,i] *dt
    end
end

function forceapplier(id1)
    x1, y1 = positiontocellcoord(id1)
    z = Int(x1 + (scaling_factor * y1))
    Threads.@threads for cell_id in lookuplist(z)
        forcefrommatrix(id1, cell_id)
    end
end

dx = 0.001

μ = .5
ϵ = .01

function forcefrommatrix(id1, cellid)
    x1, y1 = predictedpositions[1, id1], predictedpositions[2, id1]
    for i in 1:N
        matpos = Int(MegaMatrix[cellid, i])
        if matpos != 0 && matpos != id1
            #delx = x1-predictedpositions[1,matpos]
            #dely =y1-predictedpositions[2,matpos]
            #dist = sqrt((delx)^2 + (dely)^2)
            #xcomp = (k*prederivative(dist)* delx)/(dist + ϵ) 
            #ycomp = (k*prederivative(dist)* dely)/(dist + ϵ)
            
            
            dist = distance(x1, y1, matpos)
            infl_dist = influence(dist)
            xcomp = (infl_dist - influence(distance(x1 + dx, y1, matpos))) / dx
            ycomp = (infl_dist - influence(distance(x1, y1 + dx, matpos))) / dx
    
            
            xcomp += μ*(particles[3,matpos] - particles[3,id1])*viscositykernel(dist)
            ycomp += μ*(particles[4,matpos] - particles[4,id1])*viscositykernel(dist)

            dvx, dvy = k*xcomp * dt , k*ycomp * dt
            velocityupdate(id1, dvx, dvy)
            velocityupdate(matpos, -dvx, -dvy)
        end
    end
end

function positiontocellcoord(id)
    x = round(particles[1, id] / R)
    y = round(particles[2, id] / R)
    return max(1, x), max(1, y)
end
CellCounts = zeros(Int, scaling_factor^2)
function updateSpatialLookup()
    MegaMatrix .= 0
    CellCounts .= 0
    for i in 1:N
        a, b = positiontocellcoord(i)
        row = clamp(Int(scaling_factor * b + a), 1, scaling_factor^2)
        CellCounts[row] += 1
        MegaMatrix[row, CellCounts[row]] = i
    end
end

function lookuplist(v)
    return filter(x -> 1 <= x <= scaling_factor^2, [v - 1, v, v + 1, v + scaling_factor, v - scaling_factor,
                                                   v + scaling_factor + 1, v + scaling_factor - 1, v - scaling_factor + 1, v - scaling_factor - 1])
end


function evolve(nts)
    for i in 1:nts
        updateSpatialLookup()
        updatePredictions()
        
        for j in 1:N
            forceapplier(j)
            velocityupdate(j,0,-g*dt)
        end
        Threads.@threads for j in 1:N
            if abs(predictedpositions[1, j] - boxsize / 2) > boxsize / 2
                particles[3, j] *= -.9
            end
            if abs(predictedpositions[2, j] - boxsize / 2) > boxsize / 2
                particles[4, j] *= -.9
            end
        end
        Threads.@threads for j in 1:N
            positionupdate(j, particles[3, j]*dt , particles[4, j]*dt)
        end
    end
end

function addtofile(filename)
    open(filename, "a") do io
        writedlm(io, particles)
        write(io, "\n")
    end
end

@Initialize

file_path = joinpath(@__DIR__, "data\\test.dat")

function main(nts)
    for i in 1:nts
        evolve(10)
        addtofile(file_path)
    end
end
nts =800
main(nts)
