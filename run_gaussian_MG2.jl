include("AdapPDMPs.jl")
include("pdmp.jl")
include("asvar.jl")

using StatsBase # for autocor in plot_pdmp
using DataFrames, StatsPlots # to collect and plot results
using Statistics,Optim, LinearAlgebra

n_batches = 50 # number of batches in batch means both for ess_pdmp as ess.


## SETTINGS FOR EXPERIMENTS WITH INCREASING CORRELATION AND FIXED DIMENSION
dims = 50
variances = [0.5, 1, 5, 10, 15]
num = Int(dims/length(variances))
vars = repeat(variances, outer=num)
correlations = 0.3
refresh_rate = 1.0;
n_experiments = 20
discr_step_ESS = 0  # --> in continuous time
discr_step_ZZ = 0.5
discr_step_BP = 0.5
time_adaps_ZZ = 2000.0
time_adaps_BP = 2000.0
time_horizon_ZZ = 100000.0; # 10000.0
time_horizon_BP = 100000.0; # 10000.0

run_zigzag = true
run_BPS = true
# seq_probs(n) = 1/(1+log(n))
seq_probs(n) = 1/log(MathConstants.e+log(n))

preserve_volume_ZZ = false
preserve_nr_events_BP = false
run_standard_methods = true
ess_in_discrete_time = false
ess_in_continuous_time = !ess_in_discrete_time
want_plots = false

function addEntry_diag!(
    df,
    sampler,
    dim,
    ρ,
    ess,
    ess_squared_radius,
    ess_logdensity,
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
            :correlation => ρ,
            :ess_median => median(ess),
            :ess_avg => mean(ess),
            :ess_var1 => ess[1],
            :ess_var2 => ess[2],
            :ess_var3 => ess[3],
            :ess_var4 => ess[4],
            :ess_var5 => ess[5],
            :ess_025 => if (any(isnan.(ess)))
                NaN
            else
                quantile(ess, 0.25)
            end,
            :ess_075 => if (any(isnan.(ess)))
                NaN
            else
                quantile(ess, 0.75)
            end,
            :ess_min => minimum(ess),
            :ess_max => maximum(ess),
            :ess_squared_radius => ess_squared_radius,
            :ess_logdensity => ess_logdensity,
            :runtime => runtime,
            :accepted_switches => accepted_switches,
            :rejected_switches => rejected_switches,
            :time_horizon => time_horizon,
            :refresh_rate => refresh_rate,
        ),
    )
end



function experiment_diag!(
    df::DataFrame,
    dim::Int,
    correlation::Real,
    vars::Array,
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
        println("correlation = ", correlation)
        println("dimension = ", dim)
    end

    μ = zeros(dim)
    Σ = diagm(vars)
    if correlation!=0
        for i=1:dims
            for j=1:(i-1)
                Σ[i,j] = correlation *sqrt(vars[i]*vars[j])
                Σ[j,i] = Σ[i,j]
            end
        end
    end
    Σ_inv = inv(Σ)
    ∇E(x) = Σ_inv*(x-μ)
    ∂E(i,x) = dot(Σ_inv[i,:],(x-μ))
    Q = Symmetric(Σ_inv)

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

                if ess_in_discrete_time
                    (ess, ess_squared_radius) = ESS_functions_noenergy(
                        skeleton_chain,
                        discr_step_ESS,
                        time_horizon_ZZ,
                        dim,
                        n_batches = n_batches,
                    )
                    ess_logdenstat = 0
                else
                    t_skeleton = getTime(skeleton_chain)
                    x_skeleton = getPosition(skeleton_chain)
                    v_skeleton = getVelocity(skeleton_chain)
                    ess = ess_pdmp_components(
                        t_skeleton,
                        x_skeleton,
                        v_skeleton,
                        n_batches = n_batches,
                    )
                    ess_squared_radius = ess_pdmp_squared_radius(
                        t_skeleton,
                        x_skeleton,
                        v_skeleton,
                        n_batches = n_batches,
                        ellipticQ = false,
                    )
                    ess_logdenstat = 0
                end

                if (plot_trajectories && i == 1)
                    plot_pdmp(skeleton_chain, name = "ZigZag")
                end
                addEntry_diag!(
                    df,
                    "ZigZag",
                    dim,
                    correlation,
                    ess,
                    ess_squared_radius,
                    ess_logdenstat,
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
                if ess_in_discrete_time
                    (ess, ess_squared_radius) = ESS_functions_noenergy(
                        skeleton_chain,
                        discr_step_ESS,
                        time_horizon_ZZ,
                        dim,
                        n_batches = n_batches,
                    )
                    ess_logdenstat = 0
                else
                    t_skeleton = getTime(skeleton_chain)
                    x_skeleton = getPosition(skeleton_chain)
                    v_skeleton = getVelocity(skeleton_chain)
                    ess = ess_pdmp_components(
                        t_skeleton,
                        x_skeleton,
                        v_skeleton,
                        n_batches = n_batches,
                    )
                    ess_squared_radius = ess_pdmp_squared_radius(
                        t_skeleton,
                        x_skeleton,
                        v_skeleton,
                        n_batches = n_batches,
                        ellipticQ = false,
                    )
                    ess_logdenstat = 0
                end

                if (plot_trajectories && i == 1)
                    plot_pdmp(skeleton_chain, name = "Adaptive ZigZag")
                end
                addEntry_diag!(
                    df,
                    "Adaptive ZigZag (full)",
                    dim,
                    correlation,
                    ess,
                    ess_squared_radius,
                    ess_logdenstat,
                    runtime,
                    accepted_switches,
                    rejected_switches,
                    time_horizon_ZZ,
                    0.0,
                )
                if verbose
                    println("running Adaptive ZigZag (diag)...")
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
                            diag_adaptation = true,
                            updating_probs = updating_probs
                        )
                )
                if ess_in_discrete_time
                    (ess, ess_squared_radius) = ESS_functions_noenergy(
                        skeleton_chain,
                        discr_step_ESS,
                        time_horizon_ZZ,
                        dim,
                        n_batches = n_batches,
                    )
                    ess_logdenstat = 0
                else
                    t_skeleton = getTime(skeleton_chain)
                    x_skeleton = getPosition(skeleton_chain)
                    v_skeleton = getVelocity(skeleton_chain)
                    ess = ess_pdmp_components(
                        t_skeleton,
                        x_skeleton,
                        v_skeleton,
                        n_batches = n_batches,
                    )
                    ess_squared_radius = ess_pdmp_squared_radius(
                        t_skeleton,
                        x_skeleton,
                        v_skeleton,
                        n_batches = n_batches,
                        ellipticQ = false,
                    )
                    ess_logdenstat = 0
                end

                if (plot_trajectories && i == 1)
                    plot_pdmp(skeleton_chain, name = "Adaptive ZigZag (diag)")
                end
                addEntry_diag!(
                    df,
                    "Adaptive ZigZag (diag)",
                    dim,
                    correlation,
                    ess,
                    ess_squared_radius,
                    ess_logdenstat,
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
                if ess_in_discrete_time
                    (ess, ess_squared_radius) = ESS_functions_noenergy(
                        skeleton_chain,
                        discr_step_ESS,
                        time_horizon_BP,
                        dim,
                        n_batches = n_batches,
                    )
                    ess_logdenstat = 0
                else
                    t_skeleton = getTime(skeleton_chain)
                    x_skeleton = getPosition(skeleton_chain)
                    v_skeleton = getVelocity(skeleton_chain)
                    ess = ess_pdmp_components(
                        t_skeleton,
                        x_skeleton,
                        v_skeleton,
                        n_batches = n_batches,
                    )
                    ess_squared_radius = ess_pdmp_squared_radius(
                        t_skeleton,
                        x_skeleton,
                        v_skeleton,
                        n_batches = n_batches,
                        ellipticQ = false,
                    )
                    ess_logdenstat = 0
                end

                if (plot_trajectories && i == 1)
                    plot_pdmp(skeleton_chain, name = "BPS")
                end

                addEntry_diag!(
                    df,
                    "BPS",
                    dim,
                    correlation,
                    ess,
                    ess_squared_radius,
                    ess_logdenstat,
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
                            λBool = true,
                            diag_adaptation = false,
                            preserve_nr_events = preserve_nr_events_BP,
                            updating_probs = updating_probs,
                        )
                )
                if ess_in_discrete_time
                    (ess, ess_squared_radius) = ESS_functions_noenergy(
                        skeleton_chain,
                        discr_step_ESS,
                        time_horizon_BP,
                        dim,
                        n_batches = n_batches,
                    )
                    ess_logdenstat = 0
                else
                    t_skeleton = getTime(skeleton_chain)
                    x_skeleton = getPosition(skeleton_chain)
                    v_skeleton = getVelocity(skeleton_chain)
                    ess = ess_pdmp_components(
                        t_skeleton,
                        x_skeleton,
                        v_skeleton,
                        n_batches = n_batches,
                    )
                    ess_squared_radius = ess_pdmp_squared_radius(
                        t_skeleton,
                        x_skeleton,
                        v_skeleton,
                        n_batches = n_batches,
                        ellipticQ = false,
                    )
                    ess_logdenstat = 0
                end

                if (plot_trajectories && i == 1)
                    plot_pdmp(skeleton_chain, name = "Adaptive BPS (full,adap)")
                end

                addEntry_diag!(
                    df,
                    "Adaptive BPS (full,adap)",
                    dim,
                    correlation,
                    ess,
                    ess_squared_radius,
                    ess_logdenstat,
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
                            λBool = true,
                            diag_adaptation = true,
                            preserve_nr_events = preserve_nr_events_BP,
                            updating_probs = updating_probs,
                        )
                )
                if ess_in_discrete_time
                    (ess, ess_squared_radius) = ESS_functions_noenergy(
                        skeleton_chain,
                        discr_step_ESS,
                        time_horizon_BP,
                        dim,
                        n_batches = n_batches,
                    )
                    ess_logdenstat = 0
                else
                    t_skeleton = getTime(skeleton_chain)
                    x_skeleton = getPosition(skeleton_chain)
                    v_skeleton = getVelocity(skeleton_chain)
                    ess = ess_pdmp_components(
                        t_skeleton,
                        x_skeleton,
                        v_skeleton,
                        n_batches = n_batches,
                    )
                    ess_squared_radius = ess_pdmp_squared_radius(
                        t_skeleton,
                        x_skeleton,
                        v_skeleton,
                        n_batches = n_batches,
                        ellipticQ = false,
                    )
                    ess_logdenstat = 0
                end

                if (plot_trajectories && i == 1)
                    plot_pdmp(skeleton_chain, name = "Adaptive BPS (diag,adap)")
                end

                addEntry_diag!(
                    df,
                    "Adaptive BPS (diag,adap)",
                    dim,
                    correlation,
                    ess,
                    ess_squared_radius,
                    ess_logdenstat,
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
                if ess_in_discrete_time
                    (ess, ess_squared_radius) = ESS_functions_noenergy(
                        skeleton_chain,
                        discr_step_ESS,
                        time_horizon_BP,
                        dim,
                        n_batches = n_batches,
                    )
                    ess_logdenstat = 0
                else
                    t_skeleton = getTime(skeleton_chain)
                    x_skeleton = getPosition(skeleton_chain)
                    v_skeleton = getVelocity(skeleton_chain)
                    ess = ess_pdmp_components(
                        t_skeleton,
                        x_skeleton,
                        v_skeleton,
                        n_batches = n_batches,
                    )
                    ess_squared_radius = ess_pdmp_squared_radius(
                        t_skeleton,
                        x_skeleton,
                        v_skeleton,
                        n_batches = n_batches,
                        ellipticQ = false,
                    )
                    ess_logdenstat = 0
                end

                if (plot_trajectories && i == 1)
                    plot_pdmp(skeleton_chain, name = "Adaptive BPS (full,fixed)")
                end

                addEntry_diag!(
                    df,
                    "Adaptive BPS (full,fixed)",
                    dim,
                    correlation,
                    ess,
                    ess_squared_radius,
                    ess_logdenstat,
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
                            diag_adaptation = true,
                            preserve_nr_events = preserve_nr_events_BP,
                            updating_probs = updating_probs,
                        )
                )
                if ess_in_discrete_time
                    (ess, ess_squared_radius) = ESS_functions_noenergy(
                        skeleton_chain,
                        discr_step_ESS,
                        time_horizon_BP,
                        dim,
                        n_batches = n_batches,
                    )
                    ess_logdenstat = 0
                else
                    t_skeleton = getTime(skeleton_chain)
                    x_skeleton = getPosition(skeleton_chain)
                    v_skeleton = getVelocity(skeleton_chain)
                    ess = ess_pdmp_components(
                        t_skeleton,
                        x_skeleton,
                        v_skeleton,
                        n_batches = n_batches,
                    )
                    ess_squared_radius = ess_pdmp_squared_radius(
                        t_skeleton,
                        x_skeleton,
                        v_skeleton,
                        n_batches = n_batches,
                        ellipticQ = false,
                    )
                    ess_logdenstat = 0
                end

                if (plot_trajectories && i == 1)
                    plot_pdmp(skeleton_chain, name = "Adaptive BPS (diag,fixed)")
                end

                addEntry_diag!(
                    df,
                    "Adaptive BPS (diag,fixed)",
                    dim,
                    correlation,
                    ess,
                    ess_squared_radius,
                    ess_logdenstat,
                    runtime,
                    accepted_switches,
                    rejected_switches,
                    time_horizon_BP,
                    refresh_rate,
                )

                if verbose
                    println("running Adaptive BPS (only refreshment)...")
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
                            λBool = true,
                            only_refresh = true,
                            diag_adaptation = false,
                            preserve_nr_events = preserve_nr_events_BP,
                            updating_probs = updating_probs,
                        )
                )
                if ess_in_discrete_time
                    (ess, ess_squared_radius) = ESS_functions_noenergy(
                        skeleton_chain,
                        discr_step_ESS,
                        time_horizon_BP,
                        dim,
                        n_batches = n_batches,
                    )
                    ess_logdenstat = 0
                else
                    t_skeleton = getTime(skeleton_chain)
                    x_skeleton = getPosition(skeleton_chain)
                    v_skeleton = getVelocity(skeleton_chain)
                    ess = ess_pdmp_components(
                        t_skeleton,
                        x_skeleton,
                        v_skeleton,
                        n_batches = n_batches,
                    )
                    ess_squared_radius = ess_pdmp_squared_radius(
                        t_skeleton,
                        x_skeleton,
                        v_skeleton,
                        n_batches = n_batches,
                        ellipticQ = false,
                    )
                    ess_logdenstat = 0
                end

                if (plot_trajectories && i == 1)
                    plot_pdmp(skeleton_chain, name = "Adaptive BPS (off,adap)")
                end

                addEntry_diag!(
                    df,
                    "Adaptive BPS (off,adap)",
                    dim,
                    correlation,
                    ess,
                    ess_squared_radius,
                    ess_logdenstat,
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
    df[!, :avg_ess_per_sec] =
        df[!, :ess_avg] ./ (df[!, :runtime])
    df[!, :min_ess_per_sec] =
        df[!, :ess_min] ./ (df[!, :runtime] )
    df[!, :squared_radius_ess_per_sec] =
        df[!, :ess_squared_radius] ./ (df[!, :runtime] )
    df[!, :logdensity_ess_per_sec] =
        df[!, :ess_logdensity] ./ (df[!, :runtime])
end

## SETTINGS FOR LONG, DIMENSION DEPENDENT RUN
# time_horizon = 10000.0;
# refresh_rate = 0.1;
# n_experiments = 20;
# observations = 10 .^ 3;
# dims = 2 .^ (1:5);


function run_test_diag!(
    df,
    correlations,
    vars,
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
        for i = 1:n_experiments
            println("....experiment ", i)
            experiment_diag!(
                df,
                dims,
                correlations,
                vars,
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

# using ProfileView

# a quick run to compile everything

df = DataFrame(
    sampler = String[],
    dimension = Int[],
    correlation = Real[],
    time_horizon = Float64[],
    refresh_rate = Float64[],
    ess_median = Float64[],
    ess_avg = Float64[],
    ess_var1 = Float64[],
    ess_var2 = Float64[],
    ess_var3 = Float64[],
    ess_var4 = Float64[],
    ess_var5 = Float64[],
    ess_025 = Float64[],
    ess_075 = Float64[],
    ess_min = Float64[],
    ess_max = Float64[],
    ess_squared_radius = Float64[],
    ess_logdensity = Float64[],
    runtime = Float64[],
    accepted_switches = Int[],
    rejected_switches = Int[]
);

run_test_diag!(
    df,
    correlations,
    vars,
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

# @df df dotplot(:observations, :avg_ess_per_sec,group=:sampler, xaxis=:log, yaxis=:log)
# savefig("boxplot-by-observations")


# write results to disk
using CSV


#CSV.write(string("gaussian-results-diagonal-dimension", dims, "-with-refresh-", refresh_rate,"-discrstep-", discr_step_ZZ, "-timeadaps-", time_adaps_ZZ,"-discrESS-", discr_step_ESS, "-horizon-",time_horizon_ZZ, ".csv"), df)
CSV.write(string("ONLYAZZS_gaussian-results-diagonal-dimension", dims, "-with-refresh-", refresh_rate,"-discrstep-", discr_step_ZZ, "-timeadaps-", time_adaps_ZZ,"-discrESS-", discr_step_ESS, "-horizon-",time_horizon_ZZ, ".csv"), df)
