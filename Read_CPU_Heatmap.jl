save_dir = @__DIR__
mkpath(save_dir) 


include("generate.jl")

 


function generateMatrixFromFile(frame)
    output = Matrix{Float64}(undef,2,N)
    output = data[[4*frame + 1,4*frame + 2], :]
end

function optimal_image(resolution, positionMatrix, displayradius)
    displayMatrix = zeros(resolution, resolution)
    scale = resolution / boxsize

    Threads.@threads for c in 1:N
        dx = round(Int, positionMatrix[1, c] * scale)
        dy = round(Int, positionMatrix[2, c] * scale)

        for i in max(1, dx-displayradius+1):min(resolution, dx+displayradius-1)
            for j in max(1, dy-displayradius +1):min(resolution, dy+displayradius-1)
                if (dx - i)^2 + (dy - j)^2 < displayradius^2
                    displayMatrix[i, j] += 1
                end
            end
        end
    end
    return displayMatrix
end


function video_maker(savefilepath,fps,resolushun,particlesizeinpixels)
    fig = Figure()
    ax = Axis(fig[1, 1])
    hm = heatmap!(ax, optimal_image(resolushun, generateMatrixFromFile(1),particlesizeinpixels))

    output_file = joinpath(@__DIR__, savefilepath)

    total_time = @elapsed record(fig, output_file, 1:(nts-1); framerate=fps) do i
        hm[1] = optimal_image(resolushun, generateMatrixFromFile(i),particlesizeinpixels)
    end
    println("Total execution time: ", total_time, " seconds")
end
nts = 100
file_path = joinpath(@__DIR__, "data\\test.dat")

data = readdlm(file_path)


resolution = 100

video_maker("saves\\run_N=$(N)_k=$(k)_R=$(R)_resolution=$(resolution).mp4",24,resolution,4)


