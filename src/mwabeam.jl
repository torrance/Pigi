const hyperbeam = :libmwa_hyperbeam
const mwabeamhdf5 = joinpath(artifact"mwabeam", "mwa_full_embedded_element_pattern.h5")

function MWAFrame(mjd)
    epoch = Measures.Epoch(Measures.Epochs.UTC, mjd * u"d")
    location = Measures.Position(:MWA32T)
    return (epoch, location)
end

struct MWABeam
    delays::Vector{UInt32}
    amps::Vector{Float64}
    ptr::Ptr{Cvoid}
end

function MWABeam(delays::Vector{T}, amps::Vector{S}=ones(16)) where {T <: Integer, S <: Real}
    @assert length(delays) == 16 "Delays must have 16 values"
    @assert length(amps) == 16 "Amps must have 16 values"

    beamptr = Ref{Ptr{Nothing}}()
    errno = @ccall hyperbeam.new_fee_beam(mwabeamhdf5::Cstring, beamptr::Ptr{Ptr{Cvoid}})::Int

    if errno != 0
        throw(ErrorException("An error occurred loading the MWA beam"))
    end

    return MWABeam(convert(Array{UInt32}, delays), convert(Array{Float64}, amps), beamptr[])
end

function getresponse(beam::MWABeam, azel::AbstractArray{NTuple{2, Float64}}, freq::Float64; normtozenith=true, parallacticangle=true)
    N = length(azel)
    azs = map(first, azel)
    zas = map(x -> π/2 - last(x), azel)  # Convert altitude to zenith angle

    response = zeros(SMatrix{2, 2, ComplexF64, 4}, size(azel))
    mwa_latitude = Ref(ustrip(Float64, u"rad", lat(Measures.Position(:MWA32T))))

    errno = @ccall hyperbeam.calc_jones_array(
        beam.ptr::Ptr{Cvoid},
        N::UInt32,
        azs::Ptr{Float64},
        zas::Ptr{Float64},
        round(UInt32, freq)::UInt32,
        beam.delays::Ptr{UInt32},
        beam.amps::Ptr{Float64},
        length(beam.amps)::Int32,  # num_amps
        normtozenith::UInt8,
        mwa_latitude::Ptr{Float64},
        0::UInt8,  # iau_order
        response::Ptr{SMatrix{2, 2, ComplexF64, 4}},
    )::Int32

    if errno != 0
        throw(ErrorException("An error occurred calculating MWA Jones matrices"))
    end

    # Response matrices are row major. Transform to column major.
    map!(transpose, response, response)

    return response
end