module Pigi
    using PyCall
    using StaticArrays
    using Unitful: Quantity, uconvert, @u_str

    include("constants.jl")
    include("uvdatum.jl")
    include("gridspec.jl")
    include("datastore.jl")
    include("measurementset.jl")
    include("partition.jl")

end
