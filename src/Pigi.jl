module Pigi
    using PyCall
    using StaticArrays

    include("constants.jl")
    include("uvdatum.jl")
    include("datastore.jl")
    include("measurementset.jl")

end
