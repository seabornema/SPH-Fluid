using CairoMakie
using Base.Threads
using LinearAlgebra
using JLD2
using CodecZlib
using Printf
using DelimitedFiles

#  CONSTANTS boxsize must be divisible by R
N = 200  #Number of particles
boxsize = 200 #size of container
dt = 0.001 #timestep size 
k = 60000000 #pressure constant
g = 3000 #gravity constant
R = 10 # particle interaction range
μ = .1 # viscosity constant
nts =300 #number of time steps


scaling_factor = Int(boxsize / R)

save_dir = @__DIR__
mkpath(save_dir)
file_path = joinpath(@__DIR__, "data\\test.dat")
open(file_path, "w") do io
    write(io, "")
end

particles = Matrix{Float64}(undef, 4, N)
predictedpositions = Matrix{Float64}(undef, 2, N)
MegaMatrix = zeros(Int, (scaling_factor)^2 + scaling_factor, N)
CellCounts = zeros(Int, scaling_factor^2)


function Initialize()
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
    return max(0, (2 / (pi^3 * R^4)) * (R - r)^3)
end

function prederivative(r)
    return (6 / (pi^3 * R^4)) * (R - r)^2
end

function viscositykernel(r)
    return max(0, 1 / (R^6) * (R - r))
end

function distance(x, y, id)
    return sqrt((x - predictedpositions[1, id])^2 + (y - predictedpositions[2, id])^2)
end

function updatePredictions()
    Threads.@threads for i in 1:N
        predictedpositions[1, i] = particles[1, i] + particles[3, i] * dt
        predictedpositions[2, i] = particles[2, i] + particles[4, i] * dt
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
function forcefrommatrix(id1, cellid)
    x1, y1 = predictedpositions[1, id1], predictedpositions[2, id1]
    for i in 1:N
        matpos = Int(MegaMatrix[cellid, i])
        if matpos != 0 && matpos != id1
            dist = distance(x1, y1, matpos)
            infl_dist = influence(dist)
            xcomp = (infl_dist - influence(distance(x1 + dx, y1, matpos))) / dx
            ycomp = (infl_dist - influence(distance(x1, y1 + dx, matpos))) / dx

            xcomp += μ * (particles[3, matpos] - particles[3, id1]) * viscositykernel(dist)
            ycomp += μ * (particles[4, matpos] - particles[4, id1]) * viscositykernel(dist)

            dvx, dvy = k * xcomp * dt, k * ycomp * dt
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
    return filter(x -> 1 <= x <= scaling_factor^2, [
        v - 1, v, v + 1,
        v + scaling_factor, v - scaling_factor,
        v + scaling_factor + 1, v + scaling_factor - 1,
        v - scaling_factor + 1, v - scaling_factor - 1
    ])
end

function evolve(nts)
    for i in 1:nts
        updateSpatialLookup()
        updatePredictions()
        for j in 1:N
            forceapplier(j)
            velocityupdate(j, 0, -g * dt)
        end
        Threads.@threads for j in 1:N
            if abs(predictedpositions[1, j] - boxsize / 2) > boxsize / 2
                particles[3, j] *= -0.9
            end
            if abs(predictedpositions[2, j] - boxsize / 2) > boxsize / 2
                particles[4, j] *= -0.9
            end
        end
        Threads.@threads for j in 1:N
            positionupdate(j, particles[3, j] * dt, particles[4, j] * dt)
        end
    end
end

function addtofile(filename)
    open(filename, "a") do io
        writedlm(io, particles)
        write(io, "\n")
    end
end


Initialize()

function main(nts)
    for i in 1:nts
        evolve(10)
        addtofile(file_path)
    end
end


main(nts)
