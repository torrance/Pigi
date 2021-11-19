"""
# Pixel convention

- Pixels are 1 indexed up to N, where pixel centers are indexed as i.
- The continuous coordinates that belong to pixel i range from [i - 0.5, i + 0.5)
- The origin of a grid is given by N // 2 + 1 in 1-indexed coordinates, to be consistent
  with there the zeroth power component is place during fft.
"""
struct GridSpec
    Nx::Int64
    Ny::Int64
    scalelm::Float64  # The scale of pixels in lm space
    scaleuv::Float64  # Corresponding scale of pixels in uv space
end

function GridSpec(Nx, Ny, scale::Quantity)
    return GridSpec(Nx, Ny, sin(scale), 1 / (sin(scale) * Nx))
end

function GridSpec(Nx, Ny; scaleuv=nothing, scalelm=nothing)
    @assert !(scaleuv == nothing && scalelm == nothing)
    @assert scaleuv !== nothing || scalelm !== nothing
    if scalelm !== nothing
        return GridSpec(Nx, Ny, scalelm, 1 / (Nx * scalelm))
    else
        return GridSpec(Nx, Ny, 1 / (Nx * scaleuv), scaleuv)
    end
end

@inline function lambda2px(ulambda, vlambda, gridspec::GridSpec)
    return (
        ulambda / gridspec.scaleuv + gridspec.Nx ÷ 2 + 1,
        vlambda / gridspec.scaleuv + gridspec.Ny ÷ 2 + 1
    )
end

@inline function lambda2px(::Type{Int}, ulambda, vlambda, gridspec::GridSpec)
    return (
        round(Int, ulambda / gridspec.scaleuv + gridspec.Nx ÷ 2 + 1),
        round(Int, vlambda / gridspec.scaleuv + gridspec.Ny ÷ 2 + 1)
    )
end

@inline function px2lambda(upx, vpx, gridspec::GridSpec)
    return (
        (upx - gridspec.Nx ÷ 2 - 1) * gridspec.scaleuv,
        (vpx - gridspec.Ny ÷ 2 - 1) * gridspec.scaleuv
    )
end

@inline function px2sky(lpx, mpx, gridspec::GridSpec)
    return (
        (lpx - gridspec.Nx ÷ 2 - 1) * gridspec.scalelm,
        (mpx - gridspec.Ny ÷ 2 - 1) * gridspec.scalelm
    )
end