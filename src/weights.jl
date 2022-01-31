#=

# Image Weighting

We use the image weighting descriptions from https://casa.nrao.edu/docs/casaref/imager.weight.html
which defines the weight of the i'th data point, w_i as:

## Natural

    w_i = ω_i = 1 / σ_i^2
    where σ_i is the estimated noise of the visisbility sample

In practice, we take the ω_i as the row weight x the weight spectrum value from the
measurement set, and assume this has been set correctly by the observatory.

## Uniform

    w_i = ω_i / W_k
    where W_k = ∑ ω_i within the k'th UV cell

## Briggs

    w_i = ____ω_i____
          1 + W_k f^2

    where f^2 =  ___(5_*_10^-R)^2___
                 ∑_k W_k^2 / ∑_i ω_i

=#

abstract type ImageWeight end

struct Natural{T <: AbstractFloat} <: ImageWeight
    normfactor::SMatrix{2, 2, T, 4}
end

function Natural(uvdata::Vector{UVDatum{T}}, gridspec::GridSpec) where T
    normfactor = MMatrix{2, 2, T, 4}(0, 0, 0, 0)
    for uvdatum in uvdata
        upx, vpx = lambda2px(Int, uvdatum.u, uvdatum.v, gridspec)
        if 1 <= upx <= gridspec.Nx && 1 <= vpx <= gridspec.Ny
            normfactor += uvdatum.weights
        end
    end

    return Natural{T}(normfactor)
end

function (w::Natural{T})(::UVDatum{T}) where T
    return SMatrix{2, 2, T, 4}(1, 1, 1, 1) ./ w.normfactor
end

struct Uniform{T <: AbstractFloat} <: ImageWeight
    imageweights::Matrix{SMatrix{2, 2, T, 4}}
    gridspec::GridSpec
    normfactor::SMatrix{2, 2, T, 4}
end

function Uniform(uvdata::Vector{UVDatum{T}}, gridspec::GridSpec) where T
    griddedweights = makegriddedweights(uvdata, gridspec)

    imageweights = map(griddedweights) do W_k
        SMatrix{2, 2, T, 4}(
            W_k[1] > 0 ? 1 / W_k[1] : 0,
            W_k[2] > 0 ? 1 / W_k[2] : 0,
            W_k[3] > 0 ? 1 / W_k[3] : 0,
            W_k[4] > 0 ? 1 / W_k[4] : 0,
        )
    end
    normfactor = sum(x -> x[1] .* x[2], zip(imageweights, griddedweights))
    return Uniform{T}(imageweights, gridspec, normfactor)
end

function (w::Uniform{T})(uvdatum::UVDatum{T}) where T
    upx, vpx = lambda2px(Int, uvdatum.u, uvdatum.v, w.gridspec)

    if checkbounds(Bool, w.imageweights, upx, vpx)
        return w.imageweights[upx, vpx] ./ w.normfactor
    else
        return SMatrix{2, 2, T, 4}(0, 0, 0, 0)
    end
end

struct Briggs{T <: AbstractFloat} <: ImageWeight
    imageweights::Matrix{SMatrix{2, 2, T, 4}}
    gridspec::GridSpec
    normfactor::SMatrix{2, 2, T, 4}
end

function Briggs(uvdata::Vector{UVDatum{T}}, gridspec::GridSpec, robust) where T
    robust = T(robust)
    griddedweights = makegriddedweights(uvdata, gridspec)

    f2 = (5 * 10^-robust)^2 ./ (sum(x -> x.^2, griddedweights) ./ sum(griddedweights))
    imageweights = map(W_k -> 1 ./ (1 .+ W_k .* f2), griddedweights)

    normfactor = sum(x -> x[1] .* x[2], zip(imageweights, griddedweights))

    return Briggs{T}(imageweights, gridspec, normfactor)
end

function (w::Briggs{T})(uvdatum::UVDatum{T}) where T
    upx, vpx = lambda2px(Int, uvdatum.u, uvdatum.v, w.gridspec)

    if checkbounds(Bool, w.imageweights, upx, vpx)
        return w.imageweights[upx, vpx] ./ w.normfactor
    else
        return SMatrix{2, 2, T, 4}(0, 0, 0, 0)
    end
end

function makegriddedweights(uvdata::Vector{UVDatum{T}}, gridspec::GridSpec) where T
    griddedweights = zeros(SMatrix{2, 2, T, 4}, gridspec.Nx, gridspec.Ny)

    for uvdatum in uvdata
        upx, vpx = lambda2px(Int, uvdatum.u, uvdatum.v, gridspec)

        # Check if the data point lies in our grid - if it's outside it won't be gridded
        # and we can ignore it.
        if checkbounds(Bool, griddedweights, upx, vpx)
            griddedweights[upx, vpx] += uvdatum.weights
        end
    end

    return griddedweights
end

function applyweights!(workunits::Vector{WorkUnit{T}}, weighter::ImageWeight) where T
    for workunit in workunits
        applyweights!(workunit.data, weighter)
    end
end

function applyweights!(uvdata::Vector{UVDatum{T}}, weighter::ImageWeight) where T
    for (i, uvd) in enumerate(uvdata)
        uvdata[i] = UVDatum{T}(
            uvd.row,
            uvd.chan,
            uvd.u,
            uvd.v,
            uvd.w,
            uvd.weights .* weighter(uvd),
            uvd.data,
        )
    end
end
