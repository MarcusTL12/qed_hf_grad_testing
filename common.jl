include("get_matrix.jl")
include("get_dipole.jl")

const Å2B = 1.8897261245650618
const kB = 3.166811563e-6

function make_inp_func(freq, pol, coup, atoms, basis)
    function make_inp(r)
        r /= Å2B
        r = reshape(r, 3, length(r) ÷ 3)
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
    available: 1920
end memory

method
    qed-hf
end method

solver scf
    restart
    gradient threshold: 1d-10
end solver scf

qed
    modes:        1
    frequency:    {$freq}
    polarization: {$(pol[1]), $(pol[2]), $(pol[3])}
    coupling:     {$coup}
end qed

hf mean value
   dipole
end hf mean value

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
    run(`$(homedir())/eT_qed_hf_grad_print/build/eT_launch.py $(name).inp --omp $(omp) --scratch ./scratch/$(name) -ks`)
    nothing
end

function delete_scratch(name)
    if isdir("./scratch/$(name)")
        rm("./scratch/$(name)"; recursive=true)
    end
end

function make_runner_func(name, freq, pol, coup, atoms, basis, omp)
    delete_scratch(name)
    inp_func = make_inp_func(freq, pol, coup, atoms, basis)
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

function make_tot_energy_function(runner_func)
    function energy_function(r)
        runner_func(r)
        get_tot_energy(runner_func.name)
    end
end

function make_grad_func(runner_func)
    function grad_function(r)
        runner_func(r)
        get_matrix("QEDHF Molecular Gradient", runner_func.name)
    end
end

function make_e_and_grad_func(runner_func)
    function e_and_grad(r)
        runner_func(r)

        get_tot_energy(runner_func.name),
        get_matrix("QEDHF Molecular Gradient", runner_func.name)
    end
end

const atom_reg = r"[A-Z][a-z]?"

function split_atoms(atoms)
    [m.match for m in eachmatch(atom_reg, atoms)]
end

function write_xyz(filename, atoms, r)
    open(filename, "w") do io
        println(io, length(atoms), '\n')
        for (i, a) in enumerate(atoms)
            println(io, "$a    $(r[1, i]) $(r[2, i]) $(r[3, i])")
        end
    end
end
