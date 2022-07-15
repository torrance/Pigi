struct WorkUnit{T}
    u0px::Int  # pixel location of 'central' pixel, defined by FFT convention
    v0px::Int
    u0::T # Central pixel value in lambda
    v0::T
    w0::T
    subgridspec::GridSpec
    Aleft::Matrix{SMatrix{2, 2, Complex{T}, 4}}
    Aright::Matrix{SMatrix{2, 2, Complex{T}, 4}}
    data::StructVector{UVDatum{T}}
end

"""
Partitioning occurs on data that has already been coarsely partitioned based on time and
frequency. Besides partitioning based on uvw location, the only additional partitioning
that may be required at a future time is partitioning is based on baselines, which is not
yet implemented.
"""
function partition(
    uvdata::AbstractVector{UVDatum{T}},
    gridspec::GridSpec,
    subgridspec::GridSpec,
    padding::Int,
    wstep::Int,
    Aterms::Matrix{SMatrix{2, 2, Complex{T}, 4}}
) where T
    # Partition the workunits into w layers to help reduce the search space during partitioning.
    wlayers = Dict{Int, Vector{WorkUnit{T}}}()
    radius = subgridspec.Nx ÷ 2 - padding

    # Initialize Aterms to match precision of uvdata
    # In future, Aterms will have 3 dimensions, one for each antenna.
    Aterms = ifftshift(Aterms, (1, 2))
    Aleft = Aterms
    Aright = Aterms

    for uvdatum in uvdata
        upx, vpx = lambda2px(uvdatum.u, uvdatum.v, gridspec)
        w0 = wstep * fld(uvdatum.w, wstep) + wstep ÷ 2

        workunits = get!(wlayers, w0) do
            WorkUnit{T}[]
        end

        # Search through existing workunits to see if our UVDatum fits somewhere already.
        found = false
        for workunit in workunits
            # The +0.5 in this condition accounts for the off-center central pixel.
            if (
                -radius <= upx - workunit.u0px + T(0.5) <= radius &&
                -radius <= vpx - workunit.v0px + T(0.5) <= radius
            )
                push!(workunit.data, uvdatum)
                found = true
                break
            end
        end
        if found
            continue
        end

        # If we made it here, we need to create a new workunit for our UVDatum.
        u0px = round(Int, upx)
        v0px = round(Int, vpx)
        u0, v0 = px2lambda(u0px, v0px, gridspec)

        data = StructVector{UVDatum{T}}(undef, 0)
        push!(data, uvdatum)
        push!(workunits, WorkUnit{T}(
            u0px, v0px, u0, v0, w0, subgridspec, Aleft, Aright, data
        ))
    end

    return [workunit for workunits in values(wlayers) for workunit in workunits]
end

function addsubgrid!(mastergrid::AbstractMatrix, subgrid::AbstractMatrix, workunit::WorkUnit)
    u0px = workunit.u0px
    v0px = workunit.v0px
    width = workunit.subgridspec.Nx ÷ 2

    for (j, vpx) in enumerate(v0px - width:v0px + width - 1)
        if 1 <= vpx <= size(mastergrid)[2]
            for (i, upx) in enumerate(u0px - width:u0px + width - 1)
                if 1 <= upx <= size(mastergrid)[1]
                    mastergrid[upx, vpx] += subgrid[i, j]
                end
            end
        end
    end
end

function addsubgrid!(mastergrid::CuMatrix, subgrid::CuMatrix, workunit::WorkUnit)
    function _addsubgrid!(mastergrid::CuDeviceMatrix, subgrid::CuDeviceMatrix, u0px, v0px, width)
        idx = blockDim().x * (blockIdx().x - 1) + threadIdx().x

        if idx > length(subgrid)
            return nothing
        end

        xpx, ypx = Tuple(CartesianIndices(subgrid)[idx])

        upx = xpx + u0px - width
        vpx = ypx + v0px - width

        if 1 <= upx <= size(mastergrid, 1) && 1 <= vpx <= size(mastergrid, 2)
            mastergrid[upx, vpx] += subgrid[xpx, ypx]
        end

        return nothing
    end

    u0px = workunit.u0px
    v0px = workunit.v0px
    width = workunit.subgridspec.Nx ÷ 2 + 1

    kernel = @cuda launch=false _addsubgrid!(mastergrid, subgrid, u0px, v0px, width)
    config = launch_configuration(kernel.fun)
    threads = min(config.threads, length(subgrid))
    blocks = cld(length(subgrid), threads)
    kernel(mastergrid, subgrid, u0px, v0px, width; threads, blocks)
end

function extractsubgrid(mastergrid::Matrix{S}, workunit::WorkUnit{T}) where {T, S <: OutputType{T}}
    subgrid = zeros(LinearData{T}, workunit.subgridspec.Nx, workunit.subgridspec.Ny)

    u0px = workunit.u0px
    v0px = workunit.v0px
    width = workunit.subgridspec.Nx ÷ 2

    for (j, vpx) in enumerate(v0px - width:v0px + width - 1)
        if 1 <= vpx <= size(mastergrid)[2]
            for (i, upx) in enumerate(u0px - width:u0px + width - 1)
                if 1 <= upx <= size(mastergrid)[1]
                    subgrid[i, j] = mastergrid[upx, vpx]
                end
            end
        end
    end

    return subgrid
end

function extractsubgrid(mastergrid::CuMatrix{S}, workunit::WorkUnit{T}) where {T, S <: OutputType{T}}
    function _extractsubgrid!(subgrid, mastergrid, u0px, v0px, width)
        idx = blockDim().x * (blockIdx().x - 1) + threadIdx().x

        if idx > length(subgrid)
            return nothing
        end

        xpx, ypx = Tuple(CartesianIndices(subgrid)[idx])

        upx = xpx + u0px - width
        vpx = ypx + v0px - width

        if 1 <= upx <= size(mastergrid, 1) && 1 <= vpx <= size(mastergrid, 2)
            subgrid[xpx, ypx] = mastergrid[upx, vpx]
        end

        return nothing
    end

    subgrid = CUDA.zeros(LinearData{T}, workunit.subgridspec.Nx, workunit.subgridspec.Ny)

    u0px = workunit.u0px
    v0px = workunit.v0px
    width = workunit.subgridspec.Nx ÷ 2 + 1

    kernel = @cuda launch=false _extractsubgrid!(subgrid, mastergrid, u0px, v0px, width)
    config = launch_configuration(kernel.fun)
    threads = min(config.threads, length(subgrid))
    blocks = cld(length(subgrid), threads)
    kernel(subgrid, mastergrid, u0px, v0px, width; threads, blocks)

    return subgrid
end