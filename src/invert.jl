function invert(subgrids::Vector{Subgrid{T}}, gridspec::GridSpec) where T
    println("Gridding subgrids...")
    @time begin
        tasks = Task[]
        for subgrid in subgrids
            push!(tasks, Threads.@spawn begin
                grid = Pigi.gridder(subgrid)
                return grid, subgrid
            end)
        end

        mastergrids = Dict{Int, Matrix{SMatrix{2, 2, Complex{T}, 4}}}()
        for task in tasks
            # We type check the call to fetch() to achieve type stability in this function.
            grid, subgrid = fetch(task)::Tuple{Matrix{SMatrix{2, 2, Complex{T}, 4}}, Subgrid{T}}
            mastergrid = get!(mastergrids, subgrid.w0) do
                zeros(SMatrix{2, 2, Complex{T}, 4}, gridspec.Nx, gridspec.Ny)
            end

            Pigi.addsubgrid!(mastergrid, grid, subgrid)
        end
    end
    println("Done.")

    println("Used $(length(mastergrids)) w-layers during gridding")

    println("FFTing wlayers...")
    @time begin
        img = zeros(SMatrix{2, 2, Complex{T}, 4}, gridspec.Nx, gridspec.Ny)
        imgflat = reinterpret(reshape, Complex{T}, img)
        plan = plan_ifft!(imgflat, (2, 3))

        ls = fftfreq(gridspec.Nx, 1 / gridspec.scaleuv)
        ms = fftfreq(gridspec.Ny, 1 / gridspec.scaleuv)

        Threads.@threads for (w0, mastergrid) in collect(mastergrids)
            mastergridflat = reinterpret(reshape, Complex{T}, mastergrid)
            wimgflat = plan * ifftshift(mastergridflat, (2, 3))
            wimg = reinterpret(reshape, SMatrix{2, 2, Complex{T}, 4}, wimgflat)

            for (mpx, m) in enumerate(ms), (lpx, l) in enumerate(ls)
                mastergrid[lpx, mpx] = wimg[lpx, mpx] * exp(2π * 1im * w0 * ndash(l, m)) * length(wimg)
            end
        end

        for mastergrid in values(mastergrids)
            img .+= mastergrid
        end
        img = fftshift(img)
    end

    return img
end