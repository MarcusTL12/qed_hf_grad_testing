using OhMyREPL

include("../../get_matrix.jl")

function make_inp_func(atoms, basis)
    function make_inp(r)
        io = IOBuffer()

        print(
            io,
            """
system
    name: Gradient
    charge: 0
end system

do
    ground state
end do

memory
    available: 128
end memory

method
    hf
end method

solver scf
    gradient threshold: 1d-10
    max iterations: 1000
end solver scf

geometry
basis: $basis
"""
        )

        for (i, a) in enumerate(atoms)
            println(io, "    ", a, "    ", r[1, i], ' ', r[2, i], ' ', r[3, i])
        end

        println(io, "end geometry")

        String(take!(io))
    end
end

function write_inp(inp, name)
    open("$name.inp", "w") do io
        print(io, inp)
    end
end

function run_inp(name, omp)
    if isnothing(omp)
        omp = parse(Int, read("omp.txt", String))
    end
    run(`$(homedir())/eT_clean/build/eT_launch.py $(name).inp --omp $(omp) --scratch ./scratch -ks`)
    nothing
end

function delete_scratch()
    if isdir("./scratch/")
        rm("./scratch/"; recursive=true)
    end
end

function make_runner_func(name, atoms, basis, omp)
    delete_scratch()
    inp_func = make_inp_func(atoms, basis)
    function runner_func(r)
        inp = inp_func(r)
        write_inp(inp, name)
        run_inp(name, omp)
    end
end

const tot_energy_reg = r"Total energy:\ +(-?\d+\.\d+)"

function get_tot_energy(name)
    m = match(tot_energy_reg, read("$name.out", String))
    parse(Float64, m.captures[1])
end

function make_ef(rf)
    function ef(r)
        rf(r)
        get_tot_energy(rf.name)
    end
end

function find_grad2(ef, r, h, q, atm)
    dr = zeros(size(r))
    dr[q, atm] = h
    (ef(r + dr) - ef(r - dr)) / 2h / 1.8897261245650618
end

function find_grad4(ef, r, h, q, atm)
    dr = zeros(size(r))
    dr[q, atm] = h
    (-ef(r + 2dr) + 8ef(r + dr) - 8ef(r - dr) + ef(r - 2dr)) / 12h / 1.8897261245650618
end

function find_grad(ef, r, h; method=find_grad2)
    grad = zeros(size(r))
    for i in 1:size(r, 1), j in 1:size(r, 2)
        grad[i, j] = method(ef, r, h, i, j)
    end
    grad
end

const atom_reg = r"[A-Z][a-z]?"

function split_atoms(atoms)
    [m.match for m in eachmatch(atom_reg, atoms)]
end
