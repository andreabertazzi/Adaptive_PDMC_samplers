include("AdapPDMPs.jl")
include("pdmp.jl")
include("asvar.jl")

using StatsBase # for autocor in plot_pdmp
using DataFrames, StatsPlots # to collect and plot results
using Statistics,Optim, LinearAlgebra

n_batches = 50
correlation = 0.25
dist = [0,0.5,1,1.5,2,2.5]
refresh_rate = 1.0;
n_experiments = 20;
dims = 3
discr_step_ESS = 0.2
discr_step_ZZ = 0.5
discr_step_BP = 0.5
time_adaps_ZZ = 2000.0
time_adaps_BP = 10000.0
time_horizon_ZZ = 2*100000.0; # 10000.0
time_horizon_BP = 4*100000.0; # 10000.0

run_zigzag = false
run_BPS = true
seq_probs(n) = 1/log(MathConstants.e+log(n))

preserve_volume_ZZ = false
preserve_nr_events_BP = false
run_standard_methods = true
ess_in_discrete_time = false
ess_in_continuous_time = !ess_in_discrete_time
want_plots = false



function addEntry!(
    df,
    sampler,
    dim,
    dis,
    mse,
    mse_squared_radius,
    mse_logdensity,
    runtime,
    accepted_switches,
    rejected_switches,
    time_horizon,
    refresh_rate,
)

    push!(
        df,
        Dict(
            :sampler => sampler,
            :dimension => dim,
            :distance => dis,
            :mse_median => median(mse),
            :mse_avg => mean(mse),
            :mse_025 => if (any(isnan.(mse)))
                NaN
            else
                NaN
                # quantile(mse, 0.25)
            end,
            :mse_075 => if (any(isnan.(mse)))
                NaN
            else
                NaN
                # quantile(ess, 0.75)
            end,
            :mse_min => minimum(mse),
            :mse_max => maximum(mse),
            :mse_squared_radius => mse_squared_radius,
            :mse_logdensity => mse_logdensity,
            :runtime => runtime,
            :accepted_switches => accepted_switches,
            :rejected_switches => rejected_switches,
            :time_horizon => time_horizon,
            :refresh_rate => refresh_rate,
        ),
    )
end



function experiment!(
    df::DataFrame,
    dim::Int,
    dis::Real,
    time_horizon_ZZ::Float64,
    time_horizon_BP::Float64,
    discr_step_ZZ::Float64,
    discr_step_BP::Float64,
    time_adaps_ZZ::Float64,
    time_adaps_BP::Float64,
    refresh_rate::Float64;
    n_experiments = 1,
    plot_trajectories = false,
    verbose = true,
    updating_probs::Function = f(n) = 1/(1+log(n)),
)

    if verbose
        println("distance = ", dis)
        println("dimension = ", dim)
    end
    μ_1 = zeros(dim)
    μ_2 = dis*ones(dim)
    vars = ones(dim)
    Σ = correlation*ones(dim,dim) + (1-correlation)I
    Σ_inv = Σ^(-1)
    # println("$Σ_inv")
    λ = 0.5
    w = (μ_2+μ_1)/2
    v = (μ_2-μ_1)/2
    true_mean   = w
    true_radius = 0.5*(sum(μ_1.^2)+sum(vars)) + 0.5*(sum(μ_2.^2)+sum(vars))
    #actually is the mean of the two radiuses?

    # define potential and hessian bound
    α = λ *exp(-0.5*transpose(w-μ_1)*Σ_inv*(w-μ_1)) * exp(transpose(w-μ_1)*Σ_inv*w)
    β = (1-λ) *exp(-0.5*transpose(w-μ_2)*Σ_inv*(w-μ_2)) * exp(transpose(w-μ_2)*Σ_inv*w)
    U_1(x) = 0.5*transpose(x-w)*Σ_inv*(x-w)
    U_2(x) = -log(α*exp(-transpose(Σ_inv*v)*x) + β*exp(transpose(Σ_inv*v)*x))
    ∇U_1(x) = Σ_inv*(x-w)
    ∂U_1(i,x) = dot(Σ_inv[i,:],(x-w))
    m(x) = transpose(Σ_inv*v)*x
    ∇U_2(x) = (Σ_inv*v) * (α*exp(-m(x))-β*exp(m(x)))/(α*exp(-m(x))+β*exp(m(x)))
    ∂U_2(i,x) = dot(Σ_inv[i,:],v) * (α*exp(-m(x))-β*exp(m(x)))/(α*exp(-m(x))+β*exp(m(x)))
    ∇E(x) = ∇U_1(x) + ∇U_2(x)
    ∂E(i,x) = ∂U_1(i,x) + ∂U_2(i,x)
    low_bd = Σ_inv - (Σ_inv*v)*transpose(Σ_inv*v)
    eigen_lowbd = eigen(low_bd)
    eigens = eigen_lowbd.values
    println("Eigenvalues are $eigens")
    max_eigenvalue_lb = maximum(abs.(eigen_lowbd.values))
    up_bd = Σ_inv
    eigen_upbd = eigen(up_bd)
    eigenvalues_upbd = real.(eigen_upbd.values)  # avoid numerical error giving imaginary part in eigenvalues
    max_eigenvalue_ub = maximum(eigenvalues_upbd)
    hess_bound = max(max_eigenvalue_ub,max_eigenvalue_lb)
    Q = hess_bound * diagm(ones(dim))
    # hess_bound = max(1,1/4*(norm(μ_2-μ_1))^2-1)

    for i = 1:n_experiments
        if (verbose && n_experiments > 1)
            println("experiment number ", i)
        end
            if run_zigzag
                if verbose
                    println("running ZigZag...")
                end
                runtime = @elapsed (
                    (skeleton_chain, accepted_switches, rejected_switches) =
                        ZigZag(∂E, Q, time_horizon_ZZ)
                )

                (mse, mse_squared_radius) = calculate_MSE(
                    skeleton_chain,
                    discr_step_ESS,
                    time_horizon_ZZ,
                    dim,
                    true_mean,
                    true_radius,
                    n_batches = n_batches,
                )
                # println("Computed MSE is $mse")
                mse_logdenstat = 0


                if (plot_trajectories && i == 1)
                    plot_pdmp(skeleton_chain; n_samples=20000, name = "ZigZag")
                end
                addEntry!(
                    df,
                    "ZigZag",
                    dim,
                    dis,
                    mse,
                    mse_squared_radius,
                    mse_logdenstat,
                    runtime,
                    accepted_switches,
                    rejected_switches,
                    time_horizon_ZZ,
                    0.0,
                )

                if verbose
                    println("running Adaptive ZigZag...")
                end
                runtime = @elapsed (
                    (skeleton_chain, accepted_switches, rejected_switches) =
                        AdaptiveZigZag(
                            ∇E,
                            ∂E,
                            Q,
                            time_horizon_ZZ,
                            discr_step_ZZ,
                            time_adaps_ZZ;
                            updating_probs = updating_probs
                        )
                )
                (mse, mse_squared_radius) = calculate_MSE(
                    skeleton_chain,
                    discr_step_ESS,
                    time_horizon_ZZ,
                    dim,
                    true_mean,
                    true_radius,
                    n_batches = n_batches,
                )
                mse_logdenstat = 0

                if (plot_trajectories && i == 1)
                    plot_pdmp(skeleton_chain; n_samples=20000, name = "Adaptive ZigZag")
                end
                addEntry!(
                    df,
                    "Adaptive ZigZag (full)",
                    dim,
                    dis,
                    mse,
                    mse_squared_radius,
                    mse_logdenstat,
                    runtime,
                    accepted_switches,
                    rejected_switches,
                    time_horizon_ZZ,
                    0.0,
                )
            end

            if run_BPS
                if verbose
                    println("running BPS...")
                end
                runtime = @elapsed (
                    (skeleton_chain, accepted_switches, rejected_switches) =
                        BPS(∇E, Q, time_horizon_BP; refresh_rate = refresh_rate)
                )
                (mse, mse_squared_radius) = calculate_MSE(
                    skeleton_chain,
                    discr_step_ESS,
                    time_horizon_BP,
                    dim,
                    true_mean,
                    true_radius,
                    n_batches = n_batches,
                )
                mse_logdenstat = 0

                if (plot_trajectories && i == 1)
                    plot_pdmp(skeleton_chain; n_samples=10000, name = "BPS")
                end

                addEntry!(
                    df,
                    "BPS",
                    dim,
                    dis,
                    mse,
                    mse_squared_radius,
                    mse_logdenstat,
                    runtime,
                    accepted_switches,
                    rejected_switches,
                    time_horizon_BP,
                    refresh_rate,
                )

                if verbose
                    println("running Adaptive BPS...")
                end
                runtime = @elapsed (
                    (skeleton_chain, accepted_switches, rejected_switches) =
                        AdaptiveBPS(
                            ∇E,
                            Q,
                            time_horizon_BP,
                            discr_step_BP,
                            time_adaps_BP;
                            refresh_rate = refresh_rate,
                            λBool = false,
                            diag_adaptation = false,
                            preserve_nr_events = preserve_nr_events_BP,
                            updating_probs = updating_probs,
                        )
                )
                (mse, mse_squared_radius) = calculate_MSE(
                    skeleton_chain,
                    discr_step_ESS,
                    time_horizon_BP,
                    dim,
                    true_mean,
                    true_radius,
                    n_batches = n_batches,
                )
                mse_logdenstat = 0

                if (plot_trajectories && i == 1)
                    plot_pdmp(skeleton_chain; n_samples=10000, name = "Adaptive BPS (full,fixed)")
                end

                addEntry!(
                    df,
                    "Adaptive BPS (full,fixed)",
                    dim,
                    dis,
                    mse,
                    mse_squared_radius,
                    mse_logdenstat,
                    runtime,
                    accepted_switches,
                    rejected_switches,
                    time_horizon_BP,
                    refresh_rate,
                )


            end
    end
end

function postprocess!(df::DataFrame)
    df[!, :avg_mse_per_sec] =
        df[!, :mse_avg] .* (df[!, :runtime])
    df[!, :min_mse_per_sec] =
        df[!, :mse_min] .* (df[!, :runtime] )
    df[!, :squared_radius_mse_per_sec] =
        df[!, :mse_squared_radius] .* (df[!, :runtime] )
    df[!, :logdensity_mse_per_sec] =
        df[!, :mse_logdensity] .* (df[!, :runtime])
end

## SETTINGS FOR LONG, DIMENSION DEPENDENT RUN
# time_horizon = 10000.0;
# refresh_rate = 0.1;
# n_experiments = 20;
# observations = 10 .^ 3;
# dims = 2 .^ (1:5);


function run_test!(
    df,
    dist,
    dims,
    n_experiments,
    time_horizon_ZZ,
    time_horizon_BP,
    discr_step_ZZ,
    discr_step_BP,
    time_adaps_ZZ,
    time_adaps_BP,
    refresh_rate;
    verbose = true,
    plot_trajectories = false,
    updating_probs::Function = f(n) = 1/(1+log(n))
)
    for dis in dist
        println("correlation = ", correlation)
        for dim in dims
            println("..dimension = ", dim)
            for i = 1:n_experiments
                println("....experiment ", i)
                experiment!(
                    df,
                    dim,
                    dis,
                    time_horizon_ZZ,
                    time_horizon_BP,
                    discr_step_ZZ,
                    discr_step_BP,
                    time_adaps_ZZ,
                    time_adaps_BP,
                    refresh_rate;
                    n_experiments = 1,
                    plot_trajectories = plot_trajectories,
                    verbose = verbose,
                    updating_probs = updating_probs
                )
            end
        end
    end
end

# a quick run to compile everything

df = DataFrame(
    sampler = String[],
    dimension = Int[],
    distance = Real[],
    time_horizon = Float64[],
    refresh_rate = Float64[],
    mse_median = Float64[],
    mse_avg = Float64[],
    mse_025 = Float64[],
    mse_075 = Float64[],
    mse_min = Float64[],
    mse_max = Float64[],
    mse_squared_radius = Float64[],
    mse_logdensity = Float64[],
    runtime = Float64[],
    accepted_switches = Int[],
    rejected_switches = Int[]
);

run_test!(
    df,
    dist,
    dims,
    n_experiments,
    time_horizon_ZZ,
    time_horizon_BP,
    discr_step_ZZ,
    discr_step_BP,
    time_adaps_ZZ,
    time_adaps_BP,
    refresh_rate,
    verbose = true,
    plot_trajectories = want_plots,
    updating_probs = seq_probs
)

postprocess!(df)

# write results to disk
using CSV
# CSV.write(string("gaussian-mixture-MSE-by-distance-dimension-", dims, "-with-correlation-", correlation, "-with-refresh-", refresh_rate,"-discrstep-", discr_step_ZZ, "-timeadaps-", time_adaps_ZZ,"-discrESS-", discr_step_ESS, "-horizon-",time_horizon_ZZ, ".csv"), df)
