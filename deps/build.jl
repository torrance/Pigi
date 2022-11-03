using Pkg.Artifacts

# This is the path to the Artifacts.toml we will manipulate
artifact_toml = joinpath(@__DIR__, "..", "Artifacts.toml")

# Query the `Artifacts.toml` file for the hash bound to the name "mwabeam"
# (returns `nothing` if no such binding exists)
mwabeam_hash = artifact_hash("mwabeam", artifact_toml)

# create_artifact() returns the content-hash of the artifact directory once we're finished creating it
mwabeam_hash = create_artifact() do artifact_dir
    # We create the artifact by simply downloading a few files into the new artifact directory
    println("Downloading...")
    download(
        "http://ws.mwatelescope.org/static/mwa_full_embedded_element_pattern.h5",
        joinpath(artifact_dir, "mwa_full_embedded_element_pattern.h5")
    )
end

# Now bind that hash within our `Artifacts.toml`.  `force = true` means that if it already exists,
# just overwrite with the new content-hash.  Unless the source files change, we do not expect
# the content hash to change, so this should not cause unnecessary version control churn.
bind_artifact!(artifact_toml, "mwabeam", mwabeam_hash; force=true)