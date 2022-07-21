using OhMyREPL
using LinearAlgebra
using Plots
using DSP
using Statistics
using KernelDensity

include("../common.jl")

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

function write_atoms(io::IO, atoms, r, v, a)
    r /= Å2B
    v /= Å2B
    for (atm, rc, vc, ac) in zip(atoms, eachcol(r), eachcol(v), eachcol(a))
        println(io, atm, "    ", rc[1], " ", rc[2], " ", rc[3],
            " ", vc[1], " ", vc[2], " ", vc[3],
            " ", ac[1], " ", ac[2], " ", ac[3])
    end
end

function calc_kin_e(masses, v)
    sum(0.5 * m * (vc ⋅ vc) for (m, vc) in zip(masses, eachcol(v)))
end

function calculate_momentum(v, masses)
    mom = zeros(Float64, 3)

    for (i, vc) in enumerate(eachcol(v))
        mom .+= vc * masses[i]
    end

    mom
end

function do_md(io::IO, n_steps, Δt, atoms, e_grad_func, r, v=zeros(size(r));
    add_first=true, t0=0.0)
    masses = [atom_mass[a] for a in atoms]

    V, g = e_grad_func(r)
    a = get_accl(masses, g)

    tot_mom = calculate_momentum(v, masses)
    drift_v = tot_mom / sum(masses)

    for vc in eachcol(v)
        vc .-= drift_v
    end

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
        write_atoms(io, atoms, r, v, a)
    end

    for i in 1:n_steps
        println("Starting iteration $i")

        v_half = v + 0.5 * a * Δt
        r += v_half * Δt
        V, g = e_grad_func(r)
        a = get_accl(masses, g)
        v = v_half + 0.5 * a * Δt

        t += Δt

        tot_mom = calculate_momentum(v, masses)
        drift_v = tot_mom / sum(masses)

        for vc in eachcol(v)
            vc .-= drift_v
        end

        K = calc_kin_e(masses, v)

        println(io, length(atoms))
        println(io, "i = ", i, "; t = ", t, "; V = ", V, "; K = ", K,
            "; qed-freq = ", e_grad_func.runner_func.inp_func.freq,
            "; qed-pol = ", e_grad_func.runner_func.inp_func.pol,
            "; qed-coup = ", e_grad_func.runner_func.inp_func.coup,
            "; basis = ", e_grad_func.runner_func.inp_func.basis,
            "; Δt = ", Δt)
        write_atoms(io, atoms, r, v, a)

        if isfile("stop")
            println("Stopping MD!")
            break
        end
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
    Δt=nothing, freq=nothing, pol=nothing, coup=nothing, basis=nothing,
    omp=nothing, v_scale=1.0)
    atoms, r, v, t, freq_l, pol_l, coup_l, basis_l, Δt_l = get_last_conf(filename)

    v *= v_scale

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
    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, omp)

    e_grad_func = make_e_and_grad_func(rf)

    open(filename, "a") do io
        do_md(io, n_steps, Δt, atoms, e_grad_func, r, v; add_first=false, t0=t)
    end
end

function keep_temp(filename, target_temp, sim_steps, avg_steps)
    resume_md(filename, sim_steps)

    avg = get_avg_last_T(filename, avg_steps)

    while !isfile("stop")
        println("Changing temp from $avg to $target_temp")
        resume_md(filename, sim_steps; v_scale=clamp(√(target_temp / avg), 0.8, 1.2))
        avg = get_avg_last_T(filename, avg_steps)
    end
end

############ ANALYSIS ########

function get_tVK(filename)
    ts = Float64[]
    Vs = Float64[]
    Ks = Float64[]
    n_atm = 0
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
    ts, Vs, Ks, n_atm
end

function get_rv(filename)
    rs = Float64[]
    vs = Float64[]
    atoms = String[]

    n_atm = 0
    n_frames = 0

    open(filename) do io
        lines = Iterators.Stateful(eachline(io))

        while !isempty(lines)
            n_frames += 1
            n_atm = parse(Int, popfirst!(lines))
            popfirst!(lines)

            atoms = String[]

            for _ in 1:n_atm
                l = popfirst!(lines)
                ls = split(l)
                push!(atoms, ls[1])
                append!(rs, parse(Float64, n) for n in ls[2:4])
                append!(vs, parse(Float64, n) for n in ls[5:7])
            end
        end
    end

    reshape(rs, 3, n_atm, n_frames),
    reshape(vs, 3, n_atm, n_frames),
    atoms
end

function plot_tVK(filename; is=:, time=false)
    ts, Vs, Ks = get_tVK(filename)

    E0 = Vs[1] + Ks[1]
    @show E0
    Vs .-= E0

    if time
        plot(ts[is], Vs[is]; label="Potential", leg=:bottomleft)
        plot!(ts[is], Ks[is]; label="Kinetic")
        plot!(ts[is], (Vs + Ks)[is]; label="Total")
    else
        plot(Vs[is]; label="Potential", leg=:bottomleft)
        plot!(Ks[is]; label="Kinetic")
        plot!((Vs+Ks)[is]; label="Total")
    end
end

function plot_VK_overlay(filename; is=:)
    ts, Vs, Ks = get_tVK(filename)

    E0 = Vs[1] + Ks[1]
    @show E0
    Vs .-= E0

    plot(eachindex(ts)[is], -Vs[is]; label="Potential", leg=:bottomleft)
    plot!(eachindex(ts)[is], Ks[is]; label="Kinetic")
    # plot!(ts, Vs + Ks; label="Total")
end

function calculate_T_instant(Ks, N_atm)
    2Ks / (kB * 3(N_atm - 1))
end

function plot_T(filename)
    ts, _, Ks, n_atm = get_tVK(filename)

    Ts = calculate_T_instant(Ks, n_atm)

    plot(ts, Ts; ylabel="T [K]")
end

function get_avg_last_T(filename, n)
    _, _, Ks, n_atm = get_tVK(filename)
    Ts = calculate_T_instant(Ks, n_atm)
    mean(@view Ts[end-(n-1):end])
end

function compare_T(filenames)
    plot(; ylabel="T [K]", leg=:bottomright)

    for (i, filename) in enumerate(filenames)
        ts, _, Ks, n_atm = get_tVK(filename)
        Ts = calculate_T_instant(Ks, n_atm)
        plot!(ts, Ts; label="$i")
    end
    plot!()
end

function calc_T_window_avg(Ts, n)
    w = ones(Float64, n) / n

    conv(Ts, w)[1:length(Ts)]
end

function plot_T_window_avg(filename, n)
    ts, _, Ks, n_atm = get_tVK(filename)

    Ts = calculate_T_instant(Ks, n_atm)

    T_avg = calc_T_window_avg(Ts, n)

    @show last(T_avg)
    @show extrema(T_avg[end-n+1:end])

    plot(ts, T_avg; leg=false)
end

function compare_T_window_avg(filenames, n)
    plot(; ylabel="T [K]", leg=:bottomright)

    for (i, filename) in enumerate(filenames)
        ts, _, Ks, n_atm = get_tVK(filename)
        Ts = calculate_T_instant(Ks, n_atm)
        T_avg = calc_T_window_avg(Ts, n)
        plot!(ts, T_avg; label="$i")
    end
    plot!()
end

function compare_E(filenames)
    plot(; ylabel="E", leg=:bottomright)

    for (i, filename) in enumerate(filenames)
        ts, Vs, Ks, n_atm = get_tVK(filename)

        E0 = Vs[1] + Ks[1]
        Vs .-= E0

        plot!(Vs + Ks; label="$i")
    end

    plot!()
end

function calculate_total_momentum(filename)
    _, vs, atoms = get_rv(filename)
    masses = [atom_mass[a] for a in atoms]

    moms = Float64[]

    for f in 1:size(vs, 3)
        append!(moms, calculate_momentum((@view vs[:, :, f]), masses))
    end

    reshape(moms, 3, size(vs, 3))
end

function calculate_center_of_mass_conf(r, atoms)
    masses = [atom_mass[a] for a in atoms]

    com = zeros(Float64, 3)

    for (i, rc) in enumerate(eachcol(r))
        com .+= rc * masses[i]
    end

    com / sum(masses)
end

function calculate_center_of_mass(filename)
    rs, _, atoms = get_rv(filename)

    coms = Float64[]

    for f in 1:size(rs, 3)
        append!(coms, calculate_center_of_mass_conf((@view rs[:, :, f]), atoms))
    end

    reshape(coms, 3, size(rs, 3))
end

function calculate_trans_E(filename)
    _, _, atoms = get_rv(filename)
    moms = calculate_total_momentum(filename)

    mass = sum(atom_mass[a] for a in atoms)

    Es = Float64[]

    for mom in eachcol(moms)
        push!(Es, mom'mom / (2 * mass))
    end

    Es
end

function calculate_ang_mom(r, v, com, atoms)
    masses = [atom_mass[a] for a in atoms]

    am = zeros(Float64, 3)

    for (i, (rc, vc)) in enumerate(zip(eachcol(r), eachcol(v)))
        am .+= ((rc - com) × vc) * masses[i]
    end

    am
end

function calculate_tot_ang_mom(filename)
    rs, vs, atoms = get_rv(filename)
    coms = calculate_center_of_mass(filename)

    ams = Float64[]

    for f in 1:size(rs, 3)
        @views append!(ams,
            calculate_ang_mom(rs[:, :, f], vs[:, :, f], coms[:, f], atoms))
    end

    reshape(ams, 3, size(rs, 3))
end

function calc_mom_intertia(r, L, atoms)
    masses = [atom_mass[a] for a in atoms]

    I = 0.0

    for (i, rc) in enumerate(eachcol(r))
        I += masses[i] * (rc'rc - (L'rc)^2 / L'L)
    end

    I
end

function calculate_rot_energy(filename)
    rs, _, atoms = get_rv(filename)
    ams = calculate_tot_ang_mom(filename)

    Es = Float64[]

    for (f, L) in enumerate(eachcol(ams))
        I = @views calc_mom_intertia(rs[:, :, f], L, atoms)
        @views push!(Es, L'L / 2I)
    end

    Es
end

function calculate_radial_dist(r, atoms, from_atm, to_atm)
    dists = Float64[]

    for i in 1:length(atoms)
        atm1 = atoms[i]
        if atm1 == from_atm
            r1 = @view r[:, i]

            range2 = if from_atm == to_atm
                (i+1):length(atoms)
            else
                1:length(atoms)
            end

            for j in range2
                atm2 = atoms[j]
                if atm2 == to_atm
                    r2 = @view r[:, j]
                    push!(dists, norm(r1 - r2))
                end
            end
        end
    end

    dists
end

function get_last_n_radial_dist(filename, from_atm, to_atm, n, spacing=1)
    r, _, atoms = get_rv(filename)

    dists = Float64[]

    rng = 1:size(r, 3)
    rng = rng[end-(spacing*n-1):spacing:end]

    for i in rng
        append!(dists, calculate_radial_dist((@view r[:, :, i]),
            atoms, from_atm, to_atm))
    end

    dists
end

function plot_dist!(data; label="")
    xs = range(extrema(data)...; length=100)

    U = kde(data)

    plot!(xs, x -> pdf(U, x); label=label)
end

function calculate_std_dev_mass(r, atoms)
    com = calculate_center_of_mass_conf(r, atoms)

    dev = zeros(Float64, 3)

    for (rc, atm) in zip(eachcol(r), atoms)
        dev += atom_mass[atm] * (rc - com) .^ 2
    end

    sqrt.(dev)
end

function calculate_dev_from_pol_h2o(r, pol)
    devs = Float64[]

    for i in 1:3:size(r, 2)
        @views oh1 = r[:, i+1] - r[:, i]
        @views oh2 = r[:, i+2] - r[:, i]

        plane_vec = oh1 × oh2

        plane_vec /= norm(plane_vec)

        push!(devs, abs(plane_vec ⋅ pol))
    end

    devs
end

function get_last_n_dev_from_pol(filename, pol, n, spacing=1)
    r, _, atoms = get_rv(filename)

    devs = Float64[]

    rng = 1:size(r, 3)
    rng = rng[end-(spacing*n-1):spacing:end]

    for i in rng
        append!(devs, calculate_dev_from_pol_h2o((@view r[:, :, i]), pol))
    end

    devs
end

function calculate_dip_dev_from_pol_h2o(r, pol)
    devs = Float64[]

    for i in 1:3:size(r, 2)
        @views oh1 = r[:, i+1] - r[:, i]
        @views oh2 = r[:, i+2] - r[:, i]

        dip_vec = oh1 + oh2

        dip_vec /= norm(dip_vec)

        push!(devs, abs(dip_vec ⋅ pol))
    end

    devs
end

function get_last_n_dip_dev_from_pol(filename, pol, n, spacing=1)
    r, _, atoms = get_rv(filename)

    devs = Float64[]

    rng = 1:size(r, 3)
    rng = rng[end-(spacing*n-1):spacing:end]

    for i in rng
        append!(devs, calculate_dip_dev_from_pol_h2o((@view r[:, :, i]), pol))
    end

    devs
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

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 4)

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

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 4)

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

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 4)

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

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 4)

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

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 4)

    e_grad_func = make_e_and_grad_func(rf)
    open("md/many_h2o/7h2o.xyz", "w") do io
        do_md(io, 10, 20.0, atoms, e_grad_func, r)
    end
end

function test_10h2o()
    atoms = split_atoms("OHH"^10)
    basis = "cc-pvdz"
    r = Float64[
        -1.36725 -0.92663 -1.90879
        -1.74234 -0.11926 -1.45603
        -1.26991 -0.53860 -2.80044
        -3.22927 -2.38220 -0.41777
        -2.63780 -2.04056 -1.12709
        -2.57307 -2.56072 0.28293
        -2.64427 1.14824 -0.85354
        -2.42916 1.95307 -0.31878
        -3.50371 0.91709 -0.43548
        0.33397 0.18985 1.13976
        0.57476 -0.73762 0.90050
        -0.56818 0.01346 1.50704
        1.16183 1.19691 -1.34410
        1.78383 0.50993 -1.63489
        0.72519 0.74469 -0.58806
        0.58685 -2.04755 -0.37608
        0.82665 -2.88213 -0.81286
        -0.11466 -1.70341 -0.97980
        -2.14222 3.15189 0.94066
        -1.15601 3.20465 0.86100
        -2.27686 3.72975 1.71058
        -4.62220 -0.26519 0.47588
        -4.32243 -1.12357 0.08076
        -5.58760 -0.35347 0.38297
        -2.18761 -0.53766 1.97494
        -2.93282 -0.22087 1.41557
        -2.62261 -0.57164 2.84473
        0.53011 2.94997 0.77347
        0.88772 2.78719 -0.12816
        0.64677 2.04447 1.14681
    ]' * Å2B

    freq = 0.5
    pol = [0, 1, 0]
    pol = pol / norm(pol)
    coup = 0.1

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 40)

    e_grad_func = make_e_and_grad_func(rf)
    open("md/many_h2o/10h2o_0.1.xyz", "w") do io
        do_md(io, 3, 25.0, atoms, e_grad_func, r)
    end
end

function test_20h2o()
    atoms = split_atoms("OHH"^20)
    basis = "cc-pvdz"
    r = Float64[
        -4.10708 -0.59028 1.09302
        -3.35828 -0.42162 1.70466
        -4.65083 -1.22593 1.62529
        -1.55147 0.80097 1.32250
        -0.60809 1.09225 1.28322
        -1.54094 0.23115 0.51465
        -3.31957 2.84156 1.31202
        -2.69302 2.07763 1.40890
        -3.09921 3.34560 2.11670
        -5.91520 -2.39777 1.96031
        -6.29883 -3.27661 2.18525
        -6.74820 -1.97369 1.63425
        -3.51597 -1.74516 -1.30834
        -2.65476 -1.28598 -1.47388
        -3.75644 -1.26200 -0.48157
        -1.11930 -0.50871 -1.11091
        -0.35691 -1.13493 -1.03793
        -0.68763 0.24291 -1.57946
        -5.43437 -3.62526 -0.64297
        -5.04722 -2.92460 -1.21468
        -5.33602 -3.18950 0.23093
        -5.20762 1.53250 -0.20021
        -4.82975 0.71100 0.20321
        -4.82010 2.18110 0.43407
        0.93402 1.57687 0.40754
        1.65636 1.05117 -0.01379
        0.53769 1.95834 -0.41251
        2.18577 -0.20070 2.28319
        2.71239 -0.24533 1.45470
        1.70447 0.63681 2.11649
        -7.54663 0.38725 -1.00831
        -6.75914 0.90291 -0.70306
        -7.57429 0.66059 -1.94213
        -0.07584 -4.63225 -1.53491
        0.00616 -5.54496 -1.86053
        -1.06291 -4.57391 -1.45856
        -7.64513 -4.33674 0.94991
        -7.04397 -4.45802 0.18357
        -8.08645 -3.49725 0.69850
        -2.87349 2.74617 -1.45164
        -2.98052 3.06667 -0.52658
        -3.72554 2.26660 -1.53877
        2.73200 -0.29075 -0.50400
        3.45098 -0.55688 -1.10593
        2.11485 -1.05868 -0.62766
        -0.34820 2.04718 -1.99494
        -0.19156 2.53141 -2.82679
        -1.28703 2.32819 -1.82224
        -2.76085 -4.42778 -1.41518
        -2.94899 -3.46279 -1.48862
        -3.65585 -4.72904 -1.15070
        -8.22768 -1.57532 0.68895
        -7.94392 -0.93223 -0.01331
        -9.09068 -1.18096 0.91417
        0.95768 -2.60679 2.01643
        0.87012 -3.03437 2.88500
        1.38334 -1.75232 2.27407
        0.93890 -2.29545 -0.67166
        0.87353 -2.56510 0.28390
        0.67586 -3.16797 -1.07064
    ]' * Å2B

    freq = 0.5
    pol = [0, 1, 0]
    # pol = pol / norm(pol)
    coup = 0.1

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 12)

    e_grad_func = make_e_and_grad_func(rf)
    open("md/many_h2o/20h2o_0.1.xyz", "w") do io
        do_md(io, 1, 50.0, atoms, e_grad_func, r)
    end
end

function test_ethylene()
    atoms = split_atoms("CCHHHH")
    basis = "aug-cc-pvdz"
    r = Float64[
        0.0000 0.0000 0.0000
        1.3400 0.0000 0.0000
        1.8850 0.9440 0.0000
        1.8850 -0.9440 0.0000
        -0.5450 0.9440 0.0000
        -0.5450 -0.9440 0.0000
    ]' * Å2B

    freq = 0.5
    pol = [1, 0.1, 0.1]
    pol /= norm(pol)
    coup = 0.05

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 12)

    e_grad_func = make_e_and_grad_func(rf)

    open("md/matteo/ethylene.xyz", "w") do io
        do_md(io, 10, 10.0, atoms, e_grad_func, r)
    end
end
