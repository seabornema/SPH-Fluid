using CairoMakie
using DelimitedFiles
using Logging

# ───── CONFIG ─────────────────────────────────────────────────────────
# Make sure this points exactly at the file you just generated!
const DATA_PATH   = joinpath(@__DIR__, "gpu_test.dat")
const OUTPUT_PATH = joinpath(@__DIR__, "saves", "run_gpu.mp4")
const FPS         = 225
const MARKER_SIZE = 15
const BOXSIZE     = 1600f0

@info "Loading data from: $DATA_PATH"
@assert isfile(DATA_PATH) "$DATA_PATH not found—check your path!"

# ───── LOAD & PARSE ───────────────────────────────────────────────────
lines = readlines(DATA_PATH)
@info "Total lines read (including blanks): " * string(length(lines))

# drop blank lines
lines = filter(x->!isempty(strip(x)), lines)
@info "Lines after stripping blanks: " * string(length(lines))

# parse each non-blank line into a Float32 vector
parsed = [parse.(Float32, split(strip(line))) for line in lines]

# check you got what you expect
@info "First parsed line (should have N entries): " * string(parsed[1])
@info "Number of parsed rows: " * string(length(parsed))

# stack into an (4*nts)×N matrix
data = reduce(hcat, parsed)'   # parsed[i] is 1×N, so hcat→N×#lines, then '→#lines×N

# infer dimensions
const N   = size(data, 2)
const nts = size(data, 1) ÷ 4
@assert size(data,1) % 4 == 0 "Number of rows is not a multiple of 4!"
@info "Inferred N = $N; inferred nts = $nts"

# ───── HELPERS ────────────────────────────────────────────────────────
get_positions(f::Int) = @view data[(4*(f-1)+1):(4*(f-1)+2), :]
function get_velocities(f::Int)
    base = 4*(f-1)
    vx = data[base+3, :]
    vy = data[base+4, :]
    speeds = sqrt.(vx.^2 .+ vy.^2)
    mn, mx = minimum(speeds), maximum(speeds)
    (speeds .- mn) ./ (mx > mn ? mx - mn : 1f0)
end

# ───── SET UP FIGURE ─────────────────────────────────────────────────
fig = Figure(size = (600,600), backgroundcolor = :black)
ax  = Axis(fig[1,1];
           backgroundcolor = :black,
           xgridvisible    = false,
           ygridvisible    = false,
           limits          = ((0f0, BOXSIZE), (0f0, BOXSIZE)))

pos  = get_positions(1); cols = get_velocities(1)
sc   = scatter!(ax, pos[1,:], pos[2,:];
                markersize = MARKER_SIZE,
                color      = cols,
                colormap   = :plasma)

# ───── RECORD VIDEO ──────────────────────────────────────────────────
record(fig, OUTPUT_PATH, 1:nts; framerate = FPS) do f
    @info("Rendering frame $f / $nts")
    p = get_positions(f); c = get_velocities(f)
    sc[1]    = p[1,:]
    sc[2]    = p[2,:]
    sc.color = c
end

println("✅ Video saved to $OUTPUT_PATH")

