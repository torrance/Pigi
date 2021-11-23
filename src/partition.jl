struct Subgrid{T}
    u0px::Int  # pixel location of 'central' pixel, defined by FFT convention
    v0px::Int
    u0::T # Central pixel value in lambda
    v0::T
    w0::T
    subgridspec::GridSpec
    Aleft::Matrix{SMatrix{2, 2, Complex{T}, 4}}
    Aright::Matrix{SMatrix{2, 2, Complex{T}, 4}}
    data::Vector{UVDatum{T}}
end

"""
Partitioning occurs on data that has already been coarsely partitioned based on time and
frequency. Besides partitioning based on uvw location, the only additional partitioning
that may be required at a future time is partitioning is based on baselines, which is not
yet implemented.
"""
function partition(
    uvdata::Vector{UVDatum{T}},
    gridspec::GridSpec,
    subgridspec::GridSpec,
    padding::Int,
    wstep::Int,
) where T
    # Partition the subgrids into w layers to help reduce the search space during partitioning.
    wlayers = Dict{Int, Vector{Subgrid{T}}}()
    radius = subgridspec.Nx ÷ 2 - padding

    # At a future point, there will be multiple of these Aterm screens that will be
    # specified per antenna pair.
    Aleft = ones(SMatrix{2, 2, Complex{T}, 4}, subgridspec.Nx, subgridspec.Ny)
    Aright = ones(SMatrix{2, 2, Complex{T}, 4}, subgridspec.Nx, subgridspec.Ny)

    for uvdatum in uvdata
        upx, vpx = lambda2px(uvdatum.u, uvdatum.v, gridspec)
        w0 = wstep * round(Int, uvdatum.w / wstep)

        subgrids = get!(wlayers, w0) do
            Subgrid{T}[]
        end

        # Search through existing subgrids to see if our UVDatum fits somewhere already.
        found = false
        for subgrid in subgrids
            # The +0.5 in this condition accounts for the off-center central pixel.
            if (
                -radius <= upx - subgrid.u0px + T(0.5) <= radius &&
                -radius <= vpx - subgrid.v0px + T(0.5) <= radius
            )
                push!(subgrid.data, uvdatum)
                found = true
                break
            end
        end
        if found
            continue
        end

        # If we made it here, we need to create a new subgrid for our UVDatum.
        u0px = round(Int, upx)
        v0px = round(Int, vpx)
        u0, v0 = px2lambda(u0px, v0px, gridspec)

        data = UVDatum{T}[uvdatum]
        sizehint!(data, 1000)
        push!(subgrids, Subgrid{T}(
            u0px, v0px, u0, v0, w0, subgridspec, Aleft, Aright, data
        ))
    end

    subgrids = [subgrid for subgrids in values(wlayers) for subgrid in subgrids]
    occupancy = [length(subgrid.data) for subgrid in subgrids]
    println("Subgrids: $(length(subgrids)) Min/mean/median/max occupancy: $(minimum(occupancy))/$(mean(occupancy))/$(median(occupancy))/$(maximum(occupancy))")

    return subgrids
end

function addsubgrid!(
    mastergrid::Matrix{SMatrix{2, 2, T, 4}},
    grid::Matrix{SMatrix{2, 2, T, 4}},
    subgrid::Subgrid
) where {T}
    u0px = subgrid.u0px
    v0px = subgrid.v0px
    width = subgrid.subgridspec.Nx ÷ 2

    for (j, vpx) in enumerate(v0px - width:v0px + width - 1)
        if 1 <= vpx <= size(mastergrid)[2]
            for (i, upx) in enumerate(u0px - width:u0px + width - 1)
                if 1 <= upx <= size(mastergrid)[1]
                    mastergrid[upx, vpx] += grid[i, j]
                end
            end
        end
    end
end
