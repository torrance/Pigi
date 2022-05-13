refs = Dict{UInt, Any}()

struct RemoteRef{T}
    pid::Int
    oid::UInt
end

function RemoteRef(v::T) where T
    global refs
    oid = objectid(v)
    refs[oid] = v

    return RemoteRef{T}(myid(), oid)
end

function Base.getindex(r::RemoteRef{T}) where T
    @assert r.pid == myid() "A RemoteRef can only be dereferenced by the owning process"
    return refs[r.oid]::T
end

struct AssignedWorkers
    pidmap::Array{Int}  # idx => pid
    pidlocks::Dict{Int, Base.Semaphore}

    function AssignedWorkers(itr)
        pidmap = collect(itr)
        pidlocks = Dict(pid => Base.Semaphore(1) for pid in unique(pidmap))
        return new(pidmap, pidlocks)
    end
end

function pmap(f, aw::AssignedWorkers, cs...)
    function wrapped(pid, args...)
        Base.acquire(aw.pidlocks[pid])
        try
            return remotecall_fetch(f, pid, args...)
        finally
            Base.release(aw.pidlocks[pid])
        end
    end

    return asyncmap(wrapped, aw.pidmap, cs...)
end
