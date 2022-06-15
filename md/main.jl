using OhMyREPL
using LinearAlgebra
using Plots

const OMP_THREADS = 42

include("../get_matrix.jl")

const Å2B = 1.8897261245650618

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
    run(`$(homedir())/eT_qed_hf_grad_print/build/eT_launch.py $(name).inp --omp $(OMP_THREADS) --scratch ./scratch -ks`)
    nothing
end

function delete_scratch()
    if isdir("./scratch/")
        rm("./scratch/"; recursive=true)
    end
end

function make_runner_func(name, freq, pol, coup, atoms, basis)
    delete_scratch()
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

function make_e_and_grad_func(runner_func)
    function e_and_grad!(r)
        runner_func(r)

        get_tot_energy(runner_func.name),
        get_matrix("QEDHF Molecular Gradient", runner_func.name)
    end
end

const atom_reg = r"[A-Z][a-z]?"

function split_atoms(atoms)
    [m.match for m in eachmatch(atom_reg, atoms)]
end

############## MD ############

const mp = 1836

const atom_mass = Dict([
    "H" => 1.008 * mp,
    "He" => 4.0026 * mp,
    "Li" => 6.94 * mp,
    "Be" => 9.0122 * mp,
    "B" => 10.81 * mp,
    "C" => 12.011 * mp,
    "N" => 14.007 * mp,
    "O" => 15.999 * mp,
    "F" => 18.998 * mp,
    "Ne" => 20.18 * mp
])

function get_accl(masses, g)
    a = zeros(size(g))
    for (i, (c1, c2)) in enumerate(zip(eachcol(a), eachcol(g)))
        c1[:] = -c2 / masses[i]
    end
    a
end

function write_atoms(io::IO, atoms, r, v)
    r /= Å2B
    v /= Å2B
    for (a, rc, vc) in zip(atoms, eachcol(r), eachcol(v))
        println(io, a, "    ", rc[1], " ", rc[2], " ", rc[3], " ", vc[1], " ", vc[2], " ", vc[3])
    end
end

function calc_kin_e(masses, v)
    sum(0.5 * m * (vc ⋅ vc) for (m, vc) in zip(masses, eachcol(v)))
end

function do_md(io::IO, n_steps, Δt, atoms, e_grad_func, r, v=zeros(size(r)); add_first=true, t0=0.0)
    masses = [atom_mass[a] for a in atoms]

    V, g = e_grad_func(r)
    a = get_accl(masses, g)

    K = calc_kin_e(masses, v)

    t = t0

    if add_first
        println(io, length(atoms))
        println(io, "i = ", 0, "; t = ", t, "; V = ", V, "; K = ", K,
            "; qed-freq = ", e_grad_func.runner_func.inp_func.freq,
            "; qed-pol = ", e_grad_func.runner_func.inp_func.pol,
            "; qed-coup = ", e_grad_func.runner_func.inp_func.coup,
            "; basis = ", e_grad_func.runner_func.inp_func.basis,
            "; Δt = ", Δt)
        write_atoms(io, atoms, r, v)
    end

    for i in 1:n_steps
        println("Starting iteration $i")

        v_half = v + 0.5 * a * Δt
        r += v_half * Δt
        V, g = e_grad_func(r)
        a = get_accl(masses, g)
        v = v_half + 0.5 * a * Δt

        t += Δt

        K = calc_kin_e(masses, v)

        println(io, length(atoms))
        println(io, "i = ", i, "; t = ", t, "; V = ", V, "; K = ", K,
            "; qed-freq = ", e_grad_func.runner_func.inp_func.freq,
            "; qed-pol = ", e_grad_func.runner_func.inp_func.pol,
            "; qed-coup = ", e_grad_func.runner_func.inp_func.coup,
            "; basis = ", e_grad_func.runner_func.inp_func.basis,
            "; Δt = ", Δt)
        write_atoms(io, atoms, r, v)
    end
end

function get_last_conf(filename)
    t = 0.0
    freq = 0.0
    pol = ""
    coup = 0.0
    basis = ""
    Δt = 0.0

    atoms = String[]
    r = Float64[]
    v = Float64[]

    open(filename) do io
        lines = Iterators.Stateful(eachline(io))

        while !isempty(lines)
            atoms = String[]
            r = Float64[]
            v = Float64[]

            n_atm = parse(Int, popfirst!(lines))
            l = popfirst!(lines)
            ls = split(l, "; ")

            t = parse(Float64, split(ls[2], " = ")[2])
            freq = parse(Float64, split(ls[5], " = ")[2])
            pol = split(ls[6], " = ")[2]
            coup = parse(Float64, split(ls[7], " = ")[2])
            basis = split(ls[8], " = ")[2]
            Δt = parse(Float64, split(ls[9], " = ")[2])

            for _ in 1:n_atm
                l = popfirst!(lines)
                ls = split(l)
                push!(atoms, ls[1])
                append!(r, parse(Float64, rc) for rc in ls[2:4])
                append!(v, parse(Float64, vc) for vc in ls[5:7])
            end
        end
    end

    atoms,
    reshape(r, 3, length(r) ÷ 3) * Å2B,
    reshape(v, 3, length(v) ÷ 3) * Å2B,
    t,
    freq,
    eval(Meta.parse(pol)),
    coup,
    basis,
    Δt
end

function resume_md(filename, n_steps;
    Δt=nothing, freq=nothing, pol=nothing, coup=nothing, basis=nothing)
    atoms, r, v, t, freq_l, pol_l, coup_l, basis_l, Δt_l = get_last_conf(filename)

    if isnothing(freq)
        freq = freq_l
    end
    if isnothing(pol)
        pol = pol_l
    end
    if isnothing(coup)
        coup = coup_l
    end
    if isnothing(basis)
        basis = basis_l
    end
    if isnothing(Δt)
        Δt = Δt_l
    end

    pol /= norm(pol)
    rf = make_runner_func("grad", freq, pol, coup, atoms, basis)

    e_grad_func = make_e_and_grad_func(rf)

    open(filename, "a") do io
        do_md(io, n_steps, Δt, atoms, e_grad_func, r, v; add_first=false, t0=t)
    end
end

############ ANALYSIS ########

function get_tVK(filename)
    ts = Float64[]
    Vs = Float64[]
    Ks = Float64[]
    open(filename) do io
        lines = Iterators.Stateful(eachline(io))

        while !isempty(lines)
            n_atm = parse(Int, popfirst!(lines))
            l = popfirst!(lines)
            ls = split(l, "; ")
            t, V, K = [parse(Float64, split(ls[i], " = ")[2]) for i in 2:4]
            push!(ts, t)
            push!(Vs, V)
            push!(Ks, K)

            for _ in 1:n_atm
                popfirst!(lines)
            end
        end
    end
    ts, Vs, Ks
end

function plot_tVK(filename)
    ts, Vs, Ks = get_tVK(filename)

    E0 = Vs[1] + Ks[1]
    @show E0
    Vs .-= E0

    plot(ts, Vs; label="Potential")
    plot!(ts, Ks; label="Kinetic")
    plot!(ts, Vs + Ks; label="Total")
end

############ TESTS ###########

function test_h2o()
    atoms = split_atoms("OHH")
    basis = "cc-pvdz"
    r = Float64[
        0.0605815 0.999184 -0.059765
        0.0605815 -0.059765 0.999184
        2.8514e-8 -3.26305e-8 4.11651e-9
    ] * Å2B

    freq = 0.5
    pol = [0.1, 1, 0.1]
    pol = pol / norm(pol)
    coup = 0.1

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis)

    e_grad_func = make_e_and_grad_func(rf)

    open("md/h2o_anims/$(coup)_$basis.xyz", "w") do io
        do_md(io, 1000, 25.0, atoms, e_grad_func, r)
    end
end

function test_h2o_wobble()
    atoms = split_atoms("OHH")
    basis = "cc-pvdz"
    r = Float64[
        -2.08349 -1.1156 -2.36234
        1.03247 0.99867 0.357
        0.0137 -0.01814 -0.62266
    ] * Å2B

    freq = 0.5
    pol = [0.1, 1, 0.1]
    pol = pol / norm(pol)
    coup = 0.0

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis)

    e_grad_func = make_e_and_grad_func(rf)

    open("md/h2o_anims/test.xyz", "w") do io
        do_md(io, 100, 10.0, atoms, e_grad_func, r)
    end
end

function test_2h2o()
    atoms = split_atoms("OHHOHH")
    basis = "cc-pvdz"
    r = Float64[
        0.00074 -0.00014 0.00003
        0.03041 0.10161 0.97727
        -0.27764 -0.92380 -0.08655
        -0.00484 0.00131 2.75024
        0.78062 -0.00539 3.32392
        -0.68819 0.38765 3.32456
    ]' * Å2B

    freq = 0.5
    pol = [0.1, 1, 0.1]
    pol = pol / norm(pol)
    coup = 0.1

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis)

    e_grad_func = make_e_and_grad_func(rf)
    open("md/2h2o_anims/$(coup)_$basis.xyz", "w") do io
        do_md(io, 10, 10.0, atoms, e_grad_func, r)
    end
end

function test_3h2o()
    atoms = split_atoms("OHHOHHOHH")
    basis = "cc-pvdz"
    r = Float64[
        -1.99995 1.33519 0.00206
        -1.03334 1.26848 0.01524
        -2.29042 0.64850 0.62097
        1.37734 0.83744 0.08635
        2.33328 0.78267 -0.06307
        1.17761 0.05034 0.61503
        1.37730 4.83740 0.08630
        2.33330 4.78270 -0.06310
        1.17760 4.05030 0.61500
    ]' * Å2B

    freq = 0.5
    pol = [0, 1, 0]
    pol = pol / norm(pol)
    coup = 0.1

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis)

    e_grad_func = make_e_and_grad_func(rf)
    open("md/3h2o_anims/$(coup)_$basis.xyz", "w") do io
        do_md(io, 10, 10.0, atoms, e_grad_func, r)
    end
end

function test_7h2o()
    atoms = split_atoms("OHHOHHOHHOHHOHHOHHOHH")
    basis = "cc-pvdz"
    r = Float64[
        -2.76420 -0.72774 -1.45990
        -2.21757 -0.06831 -0.95981
        -2.73768 -0.31471 -2.34353
        -1.22859 1.09937 -0.13416
        -0.44342 0.54547 0.12127
        -0.82078 1.98081 -0.02962
        -5.18189 0.42640 0.26551
        -4.61686 -0.26208 -0.13692
        -6.00859 -0.04387 0.45657
        -2.11075 -3.14665 -0.38463
        -2.32780 -2.29126 -0.83165
        -2.99498 -3.54743 -0.33553
        0.49472 -2.93885 0.29290
        0.73698 -3.88059 0.29522
        -0.47329 -3.01635 0.09410
        -3.63850 2.67099 0.09481
        -2.90992 2.02140 0.11762
        -4.40376 2.06407 0.20872
        1.00864 -0.29512 0.33482
        1.98068 -0.32415 0.31304
        0.81598 -1.26755 0.37072
    ]' * Å2B

    freq = 0.5
    pol = [0, 1, 0]
    pol = pol / norm(pol)
    coup = 0.01

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis)

    e_grad_func = make_e_and_grad_func(rf)
    open("md/many_h2o/7h2o.xyz", "w") do io
        do_md(io, 10, 20.0, atoms, e_grad_func, r)
    end
end
