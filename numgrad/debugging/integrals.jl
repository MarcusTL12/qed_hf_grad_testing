
include("../main.jl")

function make_mf(rf, mat_name)
    function mf(r)
        rf(r)
        get_matrix(mat_name, rf.name)
    end
end

function make_df(rf)
    function df(r)
        rf(r)
        dx = get_matrix("AO-dipole x", rf.name)
        dy = get_matrix("AO-dipole y", rf.name)
        dz = get_matrix("AO-dipole z", rf.name)

        [dx;;; dy;;; dz]
    end
end

function make_pol_df(df, λ)
    function pol_df(r)
        d = df(r)
        pd = zeros(size(d, 1), size(d, 2))

        for i in 1:size(pd, 1), j in 1:size(pd, 2)
            @views pd[i, j] = λ ⋅ d[i, j, :]
        end

        pd
    end
end
