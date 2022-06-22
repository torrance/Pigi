# TODO: Add these as artifacts
const mwabeamhdf5 = "/home/ubuntu/mwa_full_embedded_element_pattern.h5"
const hyperbeam = "/home/ubuntu/mwa_hyperbeam/target/release/libmwa_hyperbeam.so"


function MWAFrame(mjd)
    AstropyCoords = PyCall.pyimport("astropy.coordinates")
    AstropyTime = PyCall.pyimport("astropy.time")

    obstime = AstropyTime.Time(mjd, format="mjd")
    location = AstropyCoords.EarthLocation.from_geodetic(lon="116:40:14.93", lat="-26:42:11.95", height=377.8)
    return AltAzFrame(AstropyCoords.AltAz(;location, obstime))
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
    errno = @ccall hyperbeam.new_fee_beam(mwabeamhdf5::Cstring, beamptr::Ptr{Ptr{Cvoid}}, C_NULL::Ptr)::Int

    if errno != 0
        throw(ErrorException("An error occurred loading the MWA beam"))
    end

    return MWABeam(convert(Array{UInt32}, delays), convert(Array{Float64}, amps), beamptr[])
end

function getresponse(beam::MWABeam, altaz::AbstractArray{NTuple{2, Float64}}, freq::Float64; normtozenith=true, parallacticangle=true)
    N = length(altaz)
    zas = map(x -> π/2 - first(x), altaz)  # Convert altitude to zenith angle
    azs = map(last , altaz)

    responseptr = Ref{Ptr{SMatrix{2, 2, ComplexF64, 4}}}()

    errno = @ccall hyperbeam.calc_jones_array(
        beam.ptr::Ptr{Cvoid},
        N::UInt32,
        azs::Ptr{Float64},
        zas::Ptr{Float64},
        round(UInt32, freq)::UInt32,
        beam.delays::Ptr{UInt32},
        beam.amps::Ptr{Float64},
        16::Int32,
        normtozenith::UInt8,
        parallacticangle::UInt8,
        responseptr::Ptr{Ptr{SMatrix{2, 2, ComplexF64, 4}}},
        C_NULL::Ptr{Cvoid},
    )::Int32

    if errno != 0
        throw(ErrorException("An error occurred calculating MWA Jones matrices"))
    end

    response = unsafe_wrap(Array{SMatrix{2, 2, ComplexF64, 4}, ndims(altaz)}, responseptr[], size(altaz), own=true)

    # Response matrices are row major. Transform to column major.
    map!(transpose, response, response)

    return response
end