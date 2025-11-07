save_dir = @__DIR__
mkpath(save_dir) 


include("Classic_CPU.jl")

 
function generateMatrixFromFile(frame)
    output = Matrix{Float64}(undef,2,N)
    output = data[[4*frame + 1,4*frame + 2], :]
end

function velocityfromframe(frame)
    temparray = []
    for i in 1:N
        push!(temparray, sqrt(data[4*frame + 3, i]^2 + data[4*frame + 4, i]^2))
    end
    min_vel = minimum(temparray)
    max_vel = maximum(temparray)
    normalized_vel = (temparray .- min_vel) ./ (max_vel .- min_vel)
    normalized_vel = temparray/500
    return normalized_vel
end


function video_maker2(savefilepath, fps, ballsize)
    f = Figure(backgroundcolor = :black)
    ax = Axis(f[1, 1]; xgridvisible = false, ygridvisible = false, backgroundcolor = :black)
    velocities = velocityfromframe(1)
    hm = scatter!(generateMatrixFromFile(1), markersize = ballsize, color = velocities, colormap = :plasma)

    output_file = joinpath(@__DIR__, savefilepath)

    total_time = @elapsed record(f, output_file, 1:(nts-1); framerate = fps) do i
        velocities = velocityfromframe(i)
        hm[1] = generateMatrixFromFile(i) 
        hm.color = velocities 
    end

    println("Total execution time: ", total_time, " seconds")
end


file_path = joinpath(@__DIR__, "data\\test.dat")

data = readdlm(file_path)


fps = 24
particle_display_size = 10
video_maker2("saves\\run_N=$(N)_k=$(k)_R=$(R)_Boxsize=$(boxsize)_μ=$(μ).mp4",fps,particle_display_size) #Running this requires some data in the test.dat file. Running this full script will generate that data

