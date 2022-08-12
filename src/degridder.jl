function degridder!(
    workunits::AbstractVector{WorkUnit{T}},
    grid::AbstractMatrix{S},
    subtaper::AbstractMatrix{T},
    degridop
) where {T, S <: OutputType{T}}
    wrapper = getwrapper(grid)

    # A terms are shared amongst work units, so we only have to transfer over the unique arrays.
    # We do this (awkwardly) by hashing them into an ID dictionary.
    Aterms = IdDict{AbstractMatrix{Comp2x2{T}}, wrapper{Comp2x2{T}}}()
    for workunit in workunits, Aterm in (workunit.Aleft, workunit.Aright)
        if !haskey(Aterms, Aterm)
            Aterms[Aterm] = wrapper(Aterm)
        end
    end

    Base.@sync for workunit in workunits
        # We do the ifft in the default stream, due to garbage issues the fft when run on multiple streams.
        # Once this is done, we pass off the rest of the work to a separate stream.begin
        subgrid = extractsubgrid(grid, workunit)
        fftshift!(subgrid)
        ifft!(subgrid)
        CUDA.synchronize()

        Base.@async begin
            # Apply A terms
            Aleft = Aterms[workunit.Aleft]
            Aright = Aterms[workunit.Aright]
            map!(subgrid, Aleft, subgrid, Aright, subtaper) do Aleft, subgrid, Aright, t
                return Aleft * subgrid * Aright' * t
            end

            uvdata = (
                u=wrapper(workunit.data.u),
                v=wrapper(workunit.data.v),
                w=wrapper(workunit.data.w),
                data=wrapper(workunit.data.data)
            )
            gpudft!(uvdata, (u0=workunit.u0, v0=workunit.v0, w0=workunit.w0), subgrid, workunit.subgridspec, degridop)

            copyto!(workunit.data.data, uvdata.data)
        end
    end
end

function gpudft!(uvdata, origin, subgrid, subgridspec, degridop)
    @kernel function _gpudft!(uvdata, origin, subgrid::AbstractMatrix{LinearData{T}}, subgridspec, degridop) where T
        idx = @index(Global)
        u, v, w = uvdata.u[idx], uvdata.v[idx], uvdata.w[idx]

        data = zero(LinearData{T})
        lms = fftfreq(subgridspec.Nx, T(1 / subgridspec.scaleuv))
        for (mpx, m) in enumerate(lms), (lpx, l) in enumerate(lms)
            phase = -2im * T(π) * (
                (u - origin.u0) * l +
                (v - origin.v0) * m +
                (w - origin.w0) * ndash(l, m)
            )
            data += subgrid[lpx, mpx] * exp(phase)
        end

        uvdata.data[idx] = degridop(uvdata.data[idx], data)
    end

    kernel = _gpudft!(kernelconf(subgrid)...)
    wait(
        kernel(uvdata, origin, subgrid, subgridspec, degridop; ndrange=length(uvdata.data))
    )
end

@inline function degridop_replace(_, new)
    return new
end

@inline function degridop_subtract(old, new)
    return old - new
end

@inline function degridop_add(old, new)
    return old + new
end