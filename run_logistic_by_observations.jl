include("AdapPDMPs.jl")
include("logistic.jl")
include("pdmp.jl")
# include("mala.jl")
include("asvar.jl")

using StatsBase # for autocor in plot_pdmp
using DataFrames, StatsPlots # to collect and plot results
using Statistics
using Optim

n_batches = 50 # number of batches in batch means both for ess_pdmp as ess.


## SETTINGS FOR LONG, SUBSAMPLING FOCUSED RUN
epsilon = 0.1
refresh_rate = 1.0;
n_experiments = 20;
observations = 10 .^ (3); # 10 .^ (2:5)
dims = [2,4,8,16]
discr_step_ESS = 0  # --> in continuous time
discr_step_ZZ = 0.5
discr_step_BP = 0.5
time_adaps_ZZ = 2000.0
time_adaps_BP = 2000.0
time_horizon_ZZ = 100000.0; # 10000.0
time_horizon_BP = 100000.0; # 10000.0
# time_adaps_ZZ_SS = 100.0
# time_adaps_BP_SS = 100.0
# time_horizon_ZZ_SS = 10000.0; # 10000.0
# time_horizon_BP_SS = 10000.0;
run_zigzag = false
run_BPS = true
# seq_probs(n) = 1/(1+log(n))
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
    n_observations,
    ess,
    ess_squared_radius,
    ess_logdensity,
    runtime,
    accepted_switches,
    rejected_switches,
    preprocessing_time,
    time_horizon,
    refresh_rate,
)

    push!(
        df,
        Dict(
            :sampler => sampler,
            :dimension => dim,
            :observations => n_observations,
            :ess_median => median(ess),
            :ess_avg => mean(ess),
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
            :preprocessing_time => preprocessing_time,
            :time_horizon => time_horizon,
            :refresh_rate => refresh_rate,
        ),
    )
end



function experiment!(
    df::DataFrame,
    dim::Int,
    n_observations::Int,
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
    by_dimension = false,
    updating_probs::Function = f(n) = 1/(1+log(n)),
)

    if verbose
        println("number of observations = ", n_observations)
        println("dimension = ", dim)
    end

    parameter = randn(dim)

    if epsilon == 0
        (Y, Z) = generateLogisticData(parameter, n_observations)
    else
        (Y, Z) = generateCorrelatedLogisticData(parameter, n_observations, epsilon)
    end

    Q = LogisticDominatingMatrix(Y)
    # M1 = opnorm(Q)
    # M1_factorized = vec([norm(Q[:,j], 2) for j in 1:size(Q)[1]])
    Q_zz = LogisticEntrywiseDominatingMatrix(Y)

    (E, ∇E, h_E, ∂E) = constructLogisticEnergy(Y, Z)

    preprocessing_time = @elapsed((x_ref, Σ_inv) = preprocess(E, ∇E, h_E, dim))
    # alternatively:
    # preprocessing_time = @elapsed(optim_sol = optimize(E, zeros(dim)))
    # x_ref = optim_sol.minimizer
    # println("Reference point:", x_ref)

    (∂E_ss, Q_zz_ss, ∇E_ss, h_E_ss, hessian_bound_ss) = LogisticSubsamplingTools(Y, Z)

    ∇E_ref = ∇E(x_ref)

    # in order not to have outrageous runtimes we rescale the time horizon of algorithms for which velocity does not depend on Σ
    # to Tr(Σ)^(1/2)* T. This is e.g. for Bouncy, ZigZag
    # Σ = inv(Σ_inv)
    # Σ_diag_inv = inv(Diagonal(Σ))
    # time_horizon_rescaled = time_horizon_ZZ * sqrt(sum(diag(Σ)))
    # time_horizon_ZZ = ceil(time_horizon_rescaled)
    # time_horizon_BP = time_horizon_ZZ

    for i = 1:n_experiments
        if (verbose && n_experiments > 1)
            println("experiment number ", i)
        end

        if !by_dimension # in which case running subsampled algorithms is feasible
            # run subsampled algorithms
            if run_zigzag
                if verbose
                    println("running ZigZag w/subsampling...")
                end
                runtime = @elapsed (
                    (skeleton_chain, accepted_switches, rejected_switches) =
                        ZigZagSubsampling(
                            ∂E_ss,
                            Q_zz_ss,
                            n_observations,
                            time_horizon_ZZ,
                            x_ref,
                            ∇E_ref,
                        )
                )
                # positions = discretise(skeleton_chain, discr_step_ZZ, time_horizon_ZZ, dim)
                # ess = ESS(positions, n_batches = n_batches)     # compute ess for all components
                # data_squared_radius = sum(positions.^2, dims = 1)
                # ess_squared_radius = ESS(data_squared_radius, n_batches = n_batches)[1]
                # println("ess and ess squared are", ess, "and",ess_squared_radius)
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
                    plot_pdmp(skeleton_chain, name = "ZigZag w/ subsampling")
                end
                addEntry!(
                    df,
                    "ZigZag w/subsampling",
                    dim,
                    n_observations,
                    ess,
                    ess_squared_radius,
                    ess_logdenstat,
                    runtime,
                    accepted_switches,
                    rejected_switches,
                    preprocessing_time,
                    time_horizon_ZZ,
                    0.0,
                )

                if verbose
                    println("running Adaptive ZigZag w/subsampling...")
                end
                runtime = @elapsed (
                    (skeleton_chain, accepted_switches, rejected_switches) =
                        AdaptiveZigZagSubsampling(
                            ∇E_ss,
                            ∂E_ss,
                            Q_zz_ss,
                            n_observations,
                            time_horizon_ZZ,
                            discr_step_ZZ,
                            time_adaps_ZZ;
                            x_ref = x_ref,
                            ∇E_ref = ∇E_ref,
                            excess_rate = 0.0,
                            diag_adaptation = false,
                            updating_probs = updating_probs,
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
                    plot_pdmp(skeleton_chain, name = "Adaptive ZigZag w/ subsampling")
                end
                addEntry!(
                    df,
                    "Adaptive ZigZag w/subsampling",
                    dim,
                    n_observations,
                    ess,
                    ess_squared_radius,
                    ess_logdenstat,
                    runtime,
                    accepted_switches,
                    rejected_switches,
                    preprocessing_time,
                    time_horizon_ZZ,
                    0.0,
                )
            end

            if run_BPS
                if verbose
                    println("running BPS w/subsampling...")
                end
                runtime = @elapsed (
                    (skeleton_chain, accepted_switches, rejected_switches) =
                        BPS_subsampling(
                            ∇E_ss,
                            hessian_bound_ss,
                            n_observations,
                            time_horizon_BP,
                            x_ref,
                            ∇E_ref,
                            refresh_rate = refresh_rate,
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
                    plot_pdmp(skeleton_chain, name = "BPS w/subsampling")
                end
                addEntry!(
                    df,
                    "BPS w/subsampling",
                    dim,
                    n_observations,
                    ess,
                    ess_squared_radius,
                    ess_logdenstat,
                    runtime,
                    accepted_switches,
                    rejected_switches,
                    preprocessing_time,
                    time_horizon_BP,
                    refresh_rate,
                )


                if verbose
                    println("running Adaptive BPS w/subsampling (full,adap)...")
                end
                runtime = @elapsed (
                    (skeleton_chain, accepted_switches, rejected_switches) =
                        AdaptiveBPS_subsampling(
                            ∇E_ss,
                            hessian_bound_ss,
                            n_observations,
                            time_horizon_BP,
                            discr_step_BP,
                            time_adaps_BP,
                            x_ref,
                            ∇E_ref;
                            refresh_rate = refresh_rate,
                            adapt_refresh = true,
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
                    plot_pdmp(
                        skeleton_chain,
                        name = "Adaptive BPS w/subsampling (full,adap)",
                    )
                end
                addEntry!(
                    df,
                    "Adaptive BPS w/subsampling (full,adap)",
                    dim,
                    n_observations,
                    ess,
                    ess_squared_radius,
                    ess_logdenstat,
                    runtime,
                    accepted_switches,
                    rejected_switches,
                    preprocessing_time,
                    time_horizon_BP,
                    refresh_rate,
                )

                if verbose
                    println("running Adaptive BPS w/subsampling (full,fixed)...")
                end
                runtime = @elapsed (
                    (skeleton_chain, accepted_switches, rejected_switches) =
                        AdaptiveBPS_subsampling(
                            ∇E_ss,
                            hessian_bound_ss,
                            n_observations,
                            time_horizon_BP,
                            discr_step_BP,
                            time_adaps_BP,
                            x_ref,
                            ∇E_ref;
                            refresh_rate = refresh_rate,
                            adapt_refresh = false,
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
                    plot_pdmp(
                        skeleton_chain,
                        name = "Adaptive BPS w/subsampling (full,fixed)",
                    )
                end
                addEntry!(
                    df,
                    "Adaptive BPS w/subsampling (full,fixed)",
                    dim,
                    n_observations,
                    ess,
                    ess_squared_radius,
                    ess_logdenstat,
                    runtime,
                    accepted_switches,
                    rejected_switches,
                    preprocessing_time,
                    time_horizon_BP,
                    refresh_rate;
                    updating_probs = updating_probs,
                )
            end
        end
        if run_standard_methods
            if run_zigzag
                if verbose
                    println("running ZigZag...")
                end
                runtime = @elapsed (
                    (skeleton_chain, accepted_switches, rejected_switches) =
                        ZigZag(∂E, Q_zz, time_horizon_ZZ)
                )   # WHY NOT USING Q?

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
                addEntry!(
                    df,
                    "ZigZag",
                    dim,
                    n_observations,
                    ess,
                    ess_squared_radius,
                    ess_logdenstat,
                    runtime,
                    accepted_switches,
                    rejected_switches,
                    0.0,
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
                addEntry!(
                    df,
                    "Adaptive ZigZag (full)",
                    dim,
                    n_observations,
                    ess,
                    ess_squared_radius,
                    ess_logdenstat,
                    runtime,
                    accepted_switches,
                    rejected_switches,
                    0.0,
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

                addEntry!(
                    df,
                    "BPS",
                    dim,
                    n_observations,
                    ess,
                    ess_squared_radius,
                    ess_logdenstat,
                    runtime,
                    accepted_switches,
                    rejected_switches,
                    0.0,
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

                addEntry!(
                    df,
                    "Adaptive BPS (full,adap)",
                    dim,
                    n_observations,
                    ess,
                    ess_squared_radius,
                    ess_logdenstat,
                    runtime,
                    accepted_switches,
                    rejected_switches,
                    0.0,
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

                addEntry!(
                    df,
                    "Adaptive BPS (full,fixed)",
                    dim,
                    n_observations,
                    ess,
                    ess_squared_radius,
                    ess_logdenstat,
                    runtime,
                    accepted_switches,
                    rejected_switches,
                    0.0,
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

                addEntry!(
                    df,
                    "Adaptive BPS (off,adap)",
                    dim,
                    n_observations,
                    ess,
                    ess_squared_radius,
                    ess_logdenstat,
                    runtime,
                    accepted_switches,
                    rejected_switches,
                    0.0,
                    time_horizon_BP,
                    refresh_rate,
                )
            end
        end
    end
end

function postprocess!(df::DataFrame)
    df[!, :avg_ess_per_sec] =
        df[!, :ess_avg] ./ (df[!, :runtime] + df[!, :preprocessing_time])
    df[!, :min_ess_per_sec] =
        df[!, :ess_min] ./ (df[!, :runtime] + df[!, :preprocessing_time])
    df[!, :squared_radius_ess_per_sec] =
        df[!, :ess_squared_radius] ./ (df[!, :runtime] + df[!, :preprocessing_time])
    df[!, :logdensity_ess_per_sec] =
        df[!, :ess_logdensity] ./ (df[!, :runtime] + df[!, :preprocessing_time])
end

## SETTINGS FOR LONG, DIMENSION DEPENDENT RUN
# time_horizon = 10000.0;
# refresh_rate = 0.1;
# n_experiments = 20;
# observations = 10 .^ 3;
# dims = 2 .^ (1:5);


function run_test!(
    df,
    observations,
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
    if length(dims) > 1
        by_dimension = true
    else
        by_dimension = false
    end
    for n_observations in observations
        println("n_observations = ", n_observations)
        for dim in dims
            println("..dimension = ", dim)
            for i = 1:n_experiments
                println("....experiment ", i)
                experiment!(
                    df,
                    dim,
                    n_observations,
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
                    by_dimension = by_dimension,
                    updating_probs = updating_probs
                )
            end
        end
    end
end

# using ProfileView

# a quick run to compile everything

df = DataFrame(
    sampler = String[],
    dimension = Int[],
    observations = Int[],
    time_horizon = Float64[],
    refresh_rate = Float64[],
    ess_median = Float64[],
    ess_avg = Float64[],
    ess_025 = Float64[],
    ess_075 = Float64[],
    ess_min = Float64[],
    ess_max = Float64[],
    ess_squared_radius = Float64[],
    ess_logdensity = Float64[],
    runtime = Float64[],
    accepted_switches = Int[],
    rejected_switches = Int[],
    preprocessing_time = Float64[],
);

run_test!(
    df,
    observations,
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
# run_test!(df,observations,dims,1,time_horizon_ZZ,
#     time_horizon_BP,discr_step_ZZ,discr_step_BP,time_adaps_ZZ,
#     time_adaps_BP,refresh_rate,plot_trajectories=true)

# Profile.clear()

# df = DataFrame(
#     sampler = String[],
#     dimension = Int[],
#     observations = Int[],
#     time_horizon = Float64[],
#     refresh_rate = Float64[],
#     ess_median = Float64[],
#     ess_avg = Float64[],
#     ess_025 = Float64[],
#     ess_075 = Float64[],
#     ess_min = Float64[],
#     ess_max = Float64[],
#     ess_squared_radius = Float64[],
#     runtime = Float64[],
#     preprocessing_time = Float64[],
# );
#
# run_test!(df, observations, dims, n_experiments, refresh_rate, time_horizon)
# @profview run_test!(df, observations, dims, n_experiments, refresh_rate, time_horizon)
postprocess!(df)

# @df df dotplot(:observations, :avg_ess_per_sec,group=:sampler, xaxis=:log, yaxis=:log)
# savefig("boxplot-by-observations")


# write results to disk
using CSV
# CSV.write(
#     string(
#         "results-by-observations-in-",
#         dims,
#         "-dimensions-refresh-",
#         refresh_rate,
#         "-eps-",
#         epsilon,
#         "discrstep",
#         discr_step_ZZ,
#         "timeadaps",
#         time_adaps_ZZ,
#         "discrESS",
#         discr_step_ESS,
#         "horizon",
#         time_horizon_ZZ,
#         ".csv",
#     ),
#     df,
# )
CSV.write(string("results-by-dimensions-with-", observations, "-observations-refresh-", refresh_rate, "-eps-", epsilon, "discrstep", discr_step_ZZ, "timeadaps", time_adaps_ZZ,"discrESS", discr_step_ESS, "horizon",time_horizon_ZZ, ".csv"), df)
#CSV.write(string("results-by-observations-in-", dims, "-dimensions-refresh-", refresh_rate, "-eps-", epsilon, "discrstep", discr_step_ZZ, "timeadaps", time_adaps_ZZ,"discrESS", discr_step_ESS,".csv"), df)
