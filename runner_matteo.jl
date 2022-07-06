# Script meant for editing and running as job on cluster

# include("numgrad/main.jl")
include("md/main.jl")

curfile = "md/matteo/ethylene.xyz"

@time resume_md(curfile, 30000; Δt=50.0)

