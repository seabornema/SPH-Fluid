using CUDA
using CairoMakie

# ───── Simulation parameters ─────────────────────────────────────────
N       = 2000                    # number of particles
BOXSIZE = Float32(1600)
dt      = Float32(0.005)
k       = Float32(6e7)
g       = Float32(3000)
R       = Float32(40)
μ       = Float32(2.0)
dx      = Float32(0.01)

# ───── Device arrays ────────────────────────────────────────────────
d_part = CuArray{Float32}(undef, 4, N)
d_pred = CuArray{Float32}(undef, 2, N)

# ───── Host initialization ──────────────────────────────────────────
h_part = Array{Float32}(undef, 4, N)
h_part[1:2, :] .= rand(Float32, 2, N) .* BOXSIZE/8 .+ [BOXSIZE/2-BOXSIZE/4,BOXSIZE/2-BOXSIZE/16]
h_part[3:4, :] .= 0f0
copyto!(d_part, h_part)

# ───── Helper functions ─────────────────────────────────────────────
@inline influence(r) = max(0f0, (2f0/(π^3 * R^4)) * (R - r)^3)
@inline viscositykernel(r) = max(0f0, (1f0/R^6) * (R - r))
@inline dist2(x1,y1,x2,y2) = sqrt((x1 - x2)^2 + (y1 - y2)^2)

# ───── GPU kernels ─────────────────────────────────────────────────

function kern_predict(part, pred)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if i ≤ size(part, 2)
        pred[1,i] = part[1,i] + part[3,i]*dt
        pred[2,i] = part[2,i] + part[4,i]*dt
    end
    return
end

function kern_force!(part, pred)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if i ≤ size(part, 2)
        dvx = 0f0; dvy = 0f0
        x1 = pred[1,i]; y1 = pred[2,i]

        for j in 1:N
            if j != i
                x2, y2 = pred[1,j], pred[2,j]
                r  = dist2(x1, y1, x2, y2)
                infl  = influence(r)
                gradx = (influence(dist2(x1-dx, y1, x2, y2)) - influence(dist2(x1+dx, y1, x2, y2))) / (2*dx)
                grady = (influence(dist2(x1, y1-dx, x2, y2)) - influence(dist2(x1, y1+dx, x2, y2))) / (2*dx)

                gradx += μ*(part[3,j] - part[3,i]) * viscositykernel(r)
                grady += μ*(part[4,j] - part[4,i]) * viscositykernel(r)

                dvx += k * gradx * dt
                dvy += k * grady * dt
            end
        end

        part[3,i] += dvx
        part[4,i] += dvy
    end
    return
end

function kern_integrate!(part, pred)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if i ≤ size(part, 2)
        part[4,i] -= g * dt

        cx, cy = BOXSIZE / 2, BOXSIZE / 2
        if abs(pred[1,i] - cx) > cx
            part[3,i] *= -0.9f0
        end
        if abs(pred[2,i] - cy) > cy
            part[4,i] *= -0.9f0
        end

        part[1,i] += part[3,i] * dt
        part[2,i] += part[4,i] * dt
    end
    return
end

# ───── Visualization setup ──────────────────────────────────────────
#fig = Figure(size = (700,700), backgroundcolor = :black)
#ax  = Axis(fig[1,1], limits=((0, BOXSIZE), (0, BOXSIZE)), backgroundcolor=:black)
#scatterplot = scatter!(ax, h_part[1,:], h_part[2,:], color=:cyan, markersize=2)
#display(fig)

# ───── Simulation loop ──────────────────────────────────────────────
threads, blocks = 256, cld(N, 256)

#@cuda threads=threads blocks=blocks kern_predict(d_part, d_pred)
#@cuda threads=threads blocks=blocks kern_force!(d_part, d_pred)
#@cuda threads=threads blocks=blocks kern_integrate!(d_part, d_pred)
#isplay(fig)
 
nts = 2000  # total number of steps
copyto!(h_part, d_part)
for step in 1:nts
    @cuda threads=threads blocks=blocks kern_predict(d_part, d_pred)
    @cuda threads=threads blocks=blocks kern_force!(d_part, d_pred)
    @cuda threads=threads blocks=blocks kern_integrate!(d_part, d_pred)
    
    # Update visualization
    fig = Figure(size = (700,700), backgroundcolor = :black)
    ax  = Axis(fig[1,1], limits=((0, BOXSIZE), (0, BOXSIZE)), backgroundcolor=:black)
    hpart =  Array(d_part) 
    CairoMakie.scatter!(hpart[1,:],hpart[2,:])
    display(fig)
end

println("Simulation complete.")
