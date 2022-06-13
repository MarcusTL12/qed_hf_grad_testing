using OhMyREPL
using LinearAlgebra
using Optim

include("get_matrix.jl")

function make_inp_func(freq, pol, coup, atoms, basis)
    function make_inp(r)
        r = reshape(r, 3, length(r) ÷ 3)
        io = IOBuffer()

        print(
            io,
            """
system
    name: H2O
    charge: 0
end system

do
    ground state
end do

memory
    available: 8
end memory

method
    qed-hf
end method

qed
    modes:        1
    frequency:    {$freq}
    polarization: {$(pol[1]), $(pol[2]), $(pol[3])}
    coupling:     {$coup}
end qed

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

function run_inp(name)
    run(`/home/marcus/eT_qed_hf_grad_print/build/eT_launch.py $(name).inp`)
    nothing
end

function make_runner_func(name, freq, pol, coup, atoms, basis)
    inp_func = make_inp_func(freq, pol, coup, atoms, basis)
    function runner_func(r)
        inp = inp_func(r)
        write_inp(inp, name)
        run_inp(name)
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

function find_grad(ef, r, h)
    grad = zeros(size(r))
    for i in 1:size(r, 1), j in 1:size(r, 2)
        dr = zeros(size(r))
        dr[i, j] = h
        grad[i, j] = (ef(r + dr) - ef(r - dr)) / 2h / 1.8897261245650618
    end
    grad
end

function make_e_and_grad_func(runner_func)
    function fg!(_, G, r)
        runner_func(r)

        if !isnothing(G)
            copyto!(G, get_matrix("QEDHF Molecular Gradient", runner_func.name))
        end

        get_tot_energy(runner_func.name)
    end
end

function write_xyz(filename, atoms, r)
    open(filename, "w") do io
        println(io, length(atoms), '\n')
        for (i, a) in enumerate(atoms)
            println(io, "$a    $(r[1, i]) $(r[2, i]) $(r[3, i])")
        end
    end
end

################## TESTS ##################

function test_h2o()
    atoms = "OHH"
    basis = "cc-pvdz"
    r = Float64[
        0 1 0
        0 0 1
        0 0 0
    ]

    freq = 0.5
    pol = [1, 0, 0]
    coup = 0.0

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis)

    fg! = make_e_and_grad_func(rf)

    optimize(Optim.only_fg!(fg!), r, BFGS())
end

function test_h2o2()
    atoms = "OHH"
    basis = "cc-pvdz"
    r = Float64[
        0 1 0
        0 0 1
        0 0 0
    ]

    freq = 0.5
    pol = [1, 0, 0]
    coup = 0.0

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis)

    fg! = make_e_and_grad_func(rf)

    optimize(Optim.only_fg!(fg!), r, BFGS())
end

function test_formacid()
    atoms = ["C", "O", "O", "H", "H"]
    basis = "cc-pvdz"
    r = [
        -0.51067 -0.45168 0.62859 -1.47669 1.44336
        0.91588 2.13252 0.20446 0.40755 -0.31111
        -0.0 -0.0 1.0e-5 0.0 -1.0e-5
    ]

    freq = 0.5
    pol = [0.4291825729017116, 0.8285415035462194, 0.35961270280516694]
    coup = 0.5

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis)

    fg! = make_e_and_grad_func(rf)

    optimize(Optim.only_fg!(fg!), r, BFGS())
end

function test_water_dimer()
    atoms = "OHHOHH"
    basis = "cc-pvdz"
    r = [
        0.587731 0.885978 0.889475 1.62689 1.29505 1.71486
        -0.604206 -1.50068 -0.28355 0.529524 0.390369 1.46854
        -0.88033 -0.915112 -0.0385382 1.68181 2.55717 1.59498
    ]

    freq = 0.5
    pol = [1, 0, 0]
    coup = 0.5

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis)

    fg! = make_e_and_grad_func(rf)

    optimize(Optim.only_fg!(fg!), r, BFGS())
end
