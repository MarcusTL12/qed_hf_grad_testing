using OhMyREPL
using LinearAlgebra
using Optim

const OMP_THREADS = 4

include("get_matrix.jl")
include("get_dipole.jl")

function make_inp_func(freq, pol, coup, atoms, basis)
    function make_inp(r)
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
    available: 8
end memory

method
    qed-hf
end method

solver scf
    restart
end solver scf

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
    run(`/home/marcus/eT_qed_hf_grad_print/build/eT_launch.py $(name).inp --omp $(OMP_THREADS) --scratch ./scratch -ks`)
    nothing
end

function delete_scratch()
    rm("./scratch/"; recursive=true)
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

################## FIXED CENTER ###########

function shift_by_first(r)
    r2 = r[:, 2:end]
    for i in 1:size(r2, 2)
        @views r2[:, i] -= r[:, 1]
    end
    r2
end

function pad_3zeros(r)
    hcat([0.0, 0.0, 0.0], r)
end

function fix_center_fg!(fg!)
    function fixed_fg!(e, G, r)
        rl = pad_3zeros(r)
        Gl = if !isnothing(G)
            pad_3zeros(G)
        else
            nothing
        end

        e = fg!(e, Gl, rl)
        copyto!(G, @view Gl[:, 2:end])
        e
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

function test_h2o_fixed()
    atoms = "OHH"
    basis = "cc-pvdz"
    r = Float64[
        0.0 0.563701 0.563701
        0.0 -0.755684 0.755684
        0.0 0.0 0.0
    ]
    r = shift_by_first(r)

    freq = 0.5
    pol = [0, 1, 1]
    pol = pol / norm(pol)
    coup = 0.9

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis)

    fg! = fix_center_fg!(make_e_and_grad_func(rf))

    o = optimize(Optim.only_fg!(fg!), r, BFGS())

    d = find_dipole(atoms, basis, pad_3zeros(o.minimizer))
    d = d / norm(d)
    @show d
    @show d ⋅ pol

    o
end

function test_h2o_2()
    atoms = "OHH"
    basis = "cc-pvdz"
    r = Float64[
        0.0242582 0.487871 0.487871
        0.333334 -0.467428 1.13409
        -3.36049e-14 1.37509e-14 1.74483e-14
    ]

    freq = 0.5
    pol = [1, 0.0, 0.1]
    pol = pol / norm(pol)
    @show pol
    coup = 0.3

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis)

    fg! = make_e_and_grad_func(rf)

    o = optimize(Optim.only_fg!(fg!), r, BFGS())

    d = find_dipole(atoms, basis, o.minimizer)
    d = d / norm(d)
    @show d
    @show d ⋅ pol

    o
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
    pol = pol / norm(pol)
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
    pol = pol / norm(pol)
    coup = 0.1

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis)

    fg! = make_e_and_grad_func(rf)

    o = optimize(Optim.only_fg!(fg!), r, BFGS())
    delete_scratch()

    d = find_dipole(atoms, basis, o.minimizer)
    d = d / norm(d)
    @show d
    @show d ⋅ pol

    o
end

function test_water_trimer()
    atoms = "OHHOHHOHH"
    basis = "aug-cc-pvdz"
    r = [
        -0.37112 -0.447052 -0.280472 -0.778473 -0.601694 -0.745832 -0.134667 -0.269435 -0.277074
        1.10706 0.94374 0.255976 1.99084 2.86104 2.04602 -0.743808 0.0397636 -1.47944
        2.19896 3.12571 1.78079 -0.490159 -0.811461 0.460394 0.0431765 -0.481524 -0.531083
    ]

    freq = 0.5
    pol = [1, 0, 0]
    pol = pol / norm(pol)
    coup = 0.1

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis)

    fg! = make_e_and_grad_func(rf)

    o = optimize(Optim.only_fg!(fg!), r, BFGS())

    d = find_dipole(atoms, basis, o.minimizer)
    d = d / norm(d)
    @show d
    @show d ⋅ pol

    o
end

function test_water4()
    atoms = "OHHOHHOHHOHH"
    basis = "cc-pvdz"
    r = copy([
        -5.48216 1.31999 -0.13616
        -4.92079 2.13320 -0.05848
        -6.35764 1.72787 -0.24490
        -3.40056 -0.42208 -0.17358
        -3.79913 -1.30045 -0.05402
        -4.21760 0.13904 -0.18189
        -1.66157 1.66196 -0.21751
        -0.79366 1.26773 -0.40721
        -2.22017 0.84375 -0.18604
        -3.73835 3.39534 0.00966
        -2.92409 2.83921 -0.09118
        -3.33220 4.26005 0.18855
        -5.48216 1.31999 -0.13616
        -4.92079 2.13320 -0.05848
        -6.35764 1.72787 -0.24490
        -3.40056 -0.42208 -0.17358
        -3.79913 -1.30045 -0.05402
        -4.21760 0.13904 -0.18189
        -1.66157 1.66196 -0.21751
        -0.79366 1.26773 -0.40721
        -2.22017 0.84375 -0.18604
        -3.73835 3.39534 0.00966
        -2.92409 2.83921 -0.09118
        -3.33220 4.26005 0.18855
        -5.48216 1.31999 -0.13616
        -4.92079 2.13320 -0.05848
        -6.35764 1.72787 -0.24490
        -3.40056 -0.42208 -0.17358
        -3.79913 -1.30045 -0.05402
        -4.21760 0.13904 -0.18189
        -1.66157 1.66196 -0.21751
        -0.79366 1.26773 -0.40721
        -2.22017 0.84375 -0.18604
        -3.73835 3.39534 0.00966
        -2.92409 2.83921 -0.09118
        -3.33220 4.26005 0.18855
    ]')

    freq = 0.5
    pol = [1, 0, 0]
    pol = pol / norm(pol)
    coup = 0.05

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis)

    fg! = make_e_and_grad_func(rf)

    o = optimize(Optim.only_fg!(fg!), r, BFGS())

    d = find_dipole(atoms, basis, o.minimizer)
    d = d / norm(d)
    @show d
    @show d ⋅ pol

    o
end
