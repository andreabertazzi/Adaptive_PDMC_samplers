using Statistics, LinearAlgebra, Compat, Plots, LaTeXStrings, StatsBase, StatsPlots
using BenchmarkTools, Optim

struct skeleton
  position::Array{Float64,1}
  velocity::Array{Float64,1}
  time::Float64
end

function getPosition(skele::Array{skeleton,1}; i_start::Integer=1, i_end::Integer=0)
  if i_end == 0
    i_end = length(skele)
  end
  n_samples = i_end - i_start + 1
  dim = size(skele[1].position,1)
  #position = Array{Float64,2}(undef,dim,n_samples)
  position = Vector{Vector{Float64}}(undef,0)
  for i = i_start:i_end
    push!(position,skele[i].position)
    #position[:,i] = skele[i].position
  end
  #println("position vector ", position)
  return position
end

function getVelocity(skele::Array{skeleton,1}; i_start::Integer=1, i_end::Integer=0)
  if i_end ==0
    i_end = length(skele)
  end
  n_samples = i_end - i_start + 1
  dim = size(skele[1].position,1)
  # velocity = []
  #velocity = Array{Float64,2}(undef,dim,n_samples)
  velocity = Vector{Vector{Float64}}(undef,0)
  for i = i_start:1:i_end
    #velocity[:,i] = skele[i].velocity
     push!(velocity, skele[i].velocity)
  end
  return velocity
end

function getTime(skele::Array{skeleton,1}; i_start::Integer=1, i_end::Integer=0)
  if i_end ==0
    i_end = length(skele)
  end
  #time = []
  time =  Vector{Float64}(undef, 0)
  for i = i_start:1:i_end
    push!(time, skele[i].time)
  end
  return time
end


function switch(a::Float64,b::Float64,Y::Float64)
  τ = Inf
  if b > 0
      if a < 0
          τ = sqrt(-Y/b) - a/b;
      else
        τ = sqrt((a/b)^2 - Y/b) - a/b;
      end
  elseif  b == 0
      if a > 0
        τ = -0.5*Y/a;
      else
        τ = Inf;
      end
  else #b[i] < 0
      if a <= 0
          τ = Inf;
      elseif  (a/b)^2 - Y/b > 0
          τ = - sqrt( (a/b)^2 - Y/b ) - a/b;
      else
          τ = Inf;
      end
  end
  return τ
end

function trajectory(index::Integer, μ_hat::Array{Float64,1}, μ_old::Array{Float64,1}, Σ_hat::AbstractVecOrMat{Float64}, M::AbstractVecOrMat{Float64}, x::Array{Float64,1}, n::Integer, Δt::Real, freq_adap::Integer,
                    time_adaps::Real,skel_chain::skeleton)
  t_start   = n * time_adaps;
  t         = t_start;
  t_fin     = (n+1) * time_adaps;
  n_start   = convert(Integer,n*freq_adap);
  for iter=1:1:freq_adap
    t_new = t + Δt;
    if t_new <= skel_chain[index].time
      next_pt = x[:,n_start + iter] + skel_chain[index-1].velocity*Δt;
    else
      while skel_chain[index].time<t_new && index < size(skel_chain,2)
        index +=1;
      end
      t_left = t_new - skel_chain[index-1].time;
      next_pt = skel_chain[index-1].position + skel_chain[index-1].velocity*t_left;
    end
    x[:,(n_start+iter+1)] = next_pt;
    t = t_new;
    μ_hat += 1/(n_start + iter) * (next_pt - μ_hat);
    Σ_hat += 1/(n_start + iter)*((next_pt - μ_hat)*(next_pt' - μ_old') - Σ_hat);
    μ_old = μ_hat;
    if t/Δt > d
      M = convert(Array{Float64,2},sqrt(Σ_hat));
    end
  end
  return μ_hat,Σ_hat,M,index
end

function discretise(skel_chain::AbstractArray, Δt::Float64, t_fin::Real, d::Integer)
  # Discretises the process described in skel_chain with step Δt for the interval
  # starting with the first time in skel_chain up to time t_fin (or the nearest
  # time point in the grid). The initial position is the first position in skel_chain.
  # The output is the process at discrete times, where the
  # first point in skel_chain is not considered.
  time = skel_chain[1].time
  i_skel = 1
  dim_skel = length(skel_chain)
  dim_temp = Int(round((t_fin-time)/Δt))
  temp = Array{Float64,2}(undef,d, dim_temp+1)
  temp[:,1] = skel_chain[1].position #this must be at time 0.0 or n*t_adaps
  for k = 1:dim_temp
    time += Δt
    if i_skel==dim_skel || time <= skel_chain[i_skel+1].time
      temp[:,k+1]= temp[:,k] + skel_chain[i_skel].velocity*Δt;
    else
      while ((i_skel+1) <= dim_skel) && (time > skel_chain[i_skel+1].time)
        i_skel+=1;
      end
      t_left = time - skel_chain[i_skel].time;
      if (t_left > Δt)
        temp[:,k+1]= temp[:,k] + skel_chain[i_skel].velocity*Δt;
      else
        temp[:,k+1]= skel_chain[i_skel].position + skel_chain[i_skel].velocity*t_left;
      end
    end
  end
  return temp[:,2:end]
end


function update_γ(batch::Array{Float64,2}, Σ_hat::Array{Float64,2},
                  μ_hat::AbstractArray, μ_old::AbstractArray, n_samples::Integer)
  d = size(batch,1)
  for i=1:size(batch,2)
    μ_hat += 1/(n_samples + i) * (batch[:,i]- μ_hat);
    for j=1:d
      for k=1:j
        Σ_hat[j,k] += (1/(n_samples + i)) *((batch[j,i] - μ_hat[j])*(batch[k,i] - μ_old[k]) - Σ_hat[j,k]);
      end
    end
    μ_old = copy(μ_hat);
  end
  # Σ_hat = symmetric(Σ_hat, :L)
  for j=1:d
    for k=1:(j-1)
      Σ_hat[k,j] = Σ_hat[j,k];
    end
  end
  return μ_hat,Σ_hat
end

function update_γ_diag(batch::Array{Float64,2}, Σ_hat::Array{Float64,1},
                  μ_hat::AbstractArray, μ_old::AbstractArray, n_samples::Integer)
  for i=1:size(batch,2)
    μ_hat += 1/(n_samples + i) * (batch[:,i]- μ_hat);
    Σ_hat += 1/(n_samples + i) * ((batch[:,i] - μ_hat).*(batch[:,i] - μ_old) - Σ_hat);
    μ_old = copy(μ_hat);
  end
  return μ_hat,Σ_hat
end

function plot_pdmp(skeleton_chain::Array{skeleton}; n_samples::Int = 250,name::String = "")
  nr_events = length(skeleton_chain)
  p1 = [skeleton_chain[i].position[1] for i in (nr_events-n_samples+1) : nr_events]
  p2 = [skeleton_chain[i].position[2] for i in (nr_events-n_samples+1) : nr_events]
  display(plot(p1,p2, ratio=:equal,label=name))
end

function ESS_functions(skeleton_chain::AbstractArray, EnFun::Function, Δt::Real, t_fin::Real, d::Integer; n_batches::Integer=50)
  positions = discretise(skeleton_chain, Δt, t_fin, d)
  ess = ESS(positions, n_batches = n_batches)
  data = sum(positions.^2, dims = 1)
  ess_squared_radius = ESS(data, n_batches = n_batches)[1]
  data = EnFun.(positions)
  ess_logdenstat = ESS(data, n_batches = n_batches)[1]
  # ess_logdenstat = zeros(length(ess_squared_radius))
  return (ess,ess_squared_radius,ess_logdenstat)
end

function ESS_functions_noenergy(skeleton_chain::AbstractArray,  Δt::Real, t_fin::Real, d::Integer; n_batches::Integer=50)
  positions = discretise(skeleton_chain, Δt, t_fin, d)
  ess = ESS(positions, n_batches = n_batches)
  data = sum(positions.^2, dims = 1)
  ess_squared_radius = ESS(data, n_batches = n_batches)[1]
  return (ess,ess_squared_radius)
end

function estimate_covariance(skeleton_chain::AbstractArray, discr_step::Real, T::Real, dim::Integer)
  N = size(skeleton_chain,2)
  chain = discretise(skeleton_chain, discr_step, T, dim)
  #mean_est = mean(chain, dims=2)
  cov_est = cov(chain, dims=2)
  return cov_est
end







function switchingtime(a::Float64,b::Float64,u::Float64=rand())
# generate switching time for rate of the form max(0, a + b s) + c
# under the assumptions that b > 0, c > 0
  if (b > 0)
    if (a < 0)
      return -a/b + switchingtime(0.0, b, u);
    else # a >= 0
      return -a/b + sqrt(a^2/b^2 - 2 * log(u)/b);
    end
  elseif (b == 0) # degenerate case
    if (a < 0)
      return Inf;
    else # a >= 0
      return -log(u)/a;
    end
  else # b <= 0
    if (a <= 0)
      return Inf;
    else # a > 0
      y = -log(u); t1=-a/b;
      if (y >= a * t1 + b *t1^2/2)
        return Inf;
      else
        return -a/b - sqrt(a^2/b^2 + 2 * y /b);
      end
    end
  end
end

function preprocess(E::Function, ∇E::Function, h_E::Function, dim::Int, x0::Vector{Float64} = zeros(dim))
    # minimize E (with gradient and Hessian g_E!, h_E!, respectively) to obtain x_ref
    # return x_ref and L, the Cholesky decomposition of Σ = [∇^2 E(x_ref)]^{-1} = L * L'

    # result = optimize(E,g_E!,h_E!,x0);

    ∇E! = function(storage, x)
        storage[:] = ∇E(x)
    end
    result = optimize(E,∇E!,x0);
    x_ref = Optim.minimizer(result);
    Σ_inv = h_E(x_ref);
    return (x_ref, Σ_inv);
end

function ZigZag(∂E::Function, Q::Symmetric{Float64,Matrix{Float64}}, T::Float64, x_init::Vector{Float64} = Vector{Float64}(undef,0),
  v_init::Vector{Int} = Vector{Int}(undef,0), excess_rate::Float64 = 0.0)
    # E_partial_derivative(i,x) is the i-th partial derivative of the potential E, evaluated in x
    # Q is a symmetric matrix with nonnegative entries such that |(∇^2 E(x))_{ij}| <= Q_{ij} for all x, i, j
    # T is time horizon

    dim = size(Q)[1]
    if (length(x_init) == 0 || length(v_init) == 0)
        x_init = zeros(dim)
        v_init = rand((-1,1), dim)
    end

    b = [norm(Q[:,i]) for i=1:dim];
    b = sqrt(dim)*b;
    # b = Q * ones(dim);

    t = 0.0;
    x = x_init; v = v_init;
    updateSkeleton = false;
    finished = false;
    skel_chain = skeleton[]
    push!(skel_chain,skeleton(x,v,t))
    rejected_switches = 0;
    accepted_switches = 0;
    initial_gradient = [∂E(i,x) for i in 1:dim];
    a = v .* initial_gradient

    Δt_proposed_switches = switchingtime.(a,b)
    if (excess_rate == 0.0)
        Δt_excess = Inf
    else
        Δt_excess = -log(rand())/(dim*excess_rate)
    end

    while (!finished)
        i = argmin(Δt_proposed_switches) # O(d)
        Δt_switch_proposed = Δt_proposed_switches[i]
        Δt = min(Δt_switch_proposed,Δt_excess);
        if t + Δt > T
            Δt = T - t
            finished = true
            updateSkeleton = true
        end
        x = x + v * Δt; # O(d)
        t = t + Δt;
        a = a + b * Δt; # O(d)

        if (!finished && Δt_switch_proposed < Δt_excess)
            switch_rate = v[i] * ∂E(i,x)
            proposedSwitchIntensity = a[i]
            if proposedSwitchIntensity < switch_rate
                println("ERROR: Switching rate exceeds bound.")
                println(" simulated rate: ", proposedSwitchIntensity)
                println(" actual switching rate: ", switch_rate)
                error("Switching rate exceeds bound.")
            end
            if rand() * proposedSwitchIntensity <= switch_rate
                # switch i-th component
                v[i] = -v[i]
                a[i] = -switch_rate
                updateSkeleton = true
                accepted_switches += 1
            else
                a[i] = switch_rate
                updateSkeleton = false
                rejected_switches += 1
            end
            # update refreshment time and switching time bound
            Δt_excess = Δt_excess - Δt_switch_proposed
            Δt_proposed_switches = Δt_proposed_switches .- Δt_switch_proposed
            Δt_proposed_switches[i] = switchingtime(a[i],b[i])
        elseif !finished
            # so we switch due to excess switching rate
            updateSkeleton = true
            i = rand(1:dim)
            v[i] = -v[i]
            a[i] = v[i] * ∂E(i,x)

            # update upcoming event times
            Δt_proposed_switches = Δt_proposed_switches .- Δt_excess
            Δt_excess = -log(rand())/(dim*excess_rate);
        end

        if updateSkeleton
            push!(skel_chain,skeleton(x,v,t))
            updateSkeleton = false
        end

    end
    println("ratio of accepted switches: ", accepted_switches/(accepted_switches+rejected_switches))
    println("number of proposed switches: ", accepted_switches + rejected_switches)

    return skel_chain,accepted_switches,rejected_switches

end


function ZigZagSubsampling(∂E::Function, Q::Symmetric{Float64,Matrix{Float64}}, N::Int, T::Float64, x_ref::Vector{Float64} = Vector{Float64}(undef,0), ∇E_ref::Vector{Float64} = Vector{Float64}(undef,0), x_init::Vector{Float64} = Vector{Float64}(undef,0), v_init::Vector{Int} = Vector{Int}(undef,0), excess_rate::Float64 = 0.0)
# Zig-Zag with subsampling and control variates
# E_partial_derivatives(i,l,x) gives the ∂_i E^l(x), where 1/N ∑_{l=1}^N E^l(x) = E(x), the full potential function
# Q is a symmetric matrix with nonnegative entries such that ∂_i ∂_j E^l(x) ≤ Q_{ij} for all i,j,l,x


    if (length(x_ref) == 0 || length(∇E_ref) == 0)
        controlvariates = false;
        error("ZigZagSubampling without control variates currently not supported")
    else
        controlvariates = true;
    end

    dim = size(Q)[1]
    if length(x_init) == 0
        if controlvariates
            x_init = x_ref
        else
            x_init = zeros(dim)
        end
    end
    if length(v_init) == 0
        v_init = rand((-1,1), dim)
    end
    C = vec(sqrt.(sum(Q.^2, dims=2)))
    b = C * sqrt(dim); #  = C |v|

    t = 0.0;
    x = x_init; v = v_init;
    updateSkeleton = false;
    finished = false;
    skel_chain = skeleton[]
    push!(skel_chain,skeleton(x,v,t))
    rejected_switches = 0;
    accepted_switches = 0;
    a = vec(v .* ∇E_ref + C * norm(x-x_ref))
    Δt_proposed_switches = switchingtime.(a,b)  #WHY USE THIS FUNCTION? SINCE a,b\geq 0 it is the same!

    if (excess_rate == 0.0)
        Δt_excess = Inf
    else
        Δt_excess = -log(rand())/(dim*excess_rate)
    end

    while (!finished)
        i = argmin(Δt_proposed_switches) # O(d)
        Δt_switch_proposed = Δt_proposed_switches[i]
        Δt = min(Δt_switch_proposed,Δt_excess);
        if t + Δt > T
            Δt = T - t
            finished = true
            updateSkeleton = true
        end
        x = x + v * Δt; # O(d)
        t = t + Δt;
        a = a + b * Δt; # O(d)

        if (!finished && Δt_switch_proposed < Δt_excess)
            j = rand(1:N)
            switch_rate = v[i] * (∇E_ref[i] + ∂E(i,j,x) - ∂E(i,j,x_ref))
            proposedSwitchIntensity = a[i]
            if proposedSwitchIntensity < switch_rate
                println("ERROR: Switching rate exceeds bound.")
                println(" simulated rate: ", proposedSwitchIntensity)
                println(" actual switching rate: ", switch_rate)
                error("Switching rate exceeds bound.")
            end
            if rand() * proposedSwitchIntensity <= switch_rate
                # switch i-th component
                v[i] = -v[i]
                a[i] = v[i] * ∇E_ref[i] + C[i] * norm(x-x_ref)
                updateSkeleton = true
                accepted_switches += 1
            else
                updateSkeleton = false
                rejected_switches += 1
            end
            # update refreshment time and switching time bound
            Δt_excess = Δt_excess - Δt_switch_proposed
            Δt_proposed_switches = Δt_proposed_switches .- Δt_switch_proposed
            Δt_proposed_switches[i] = switchingtime(a[i],b[i])
        elseif !finished
            # so we switch due to excess switching rate
            updateSkeleton = true
            i = rand(1:dim)
            v[i] = -v[i]
            a[i] = v[i] * ∇E_ref[i] + C[i] * norm(x-x_ref)

            # update upcoming event times
            Δt_proposed_switches = Δt_proposed_switches .- Δt_excess
            Δt_excess = -log(rand())/(dim*excess_rate);
        end

        if updateSkeleton
            push!(skel_chain,skeleton(x,v,t))
            updateSkeleton = false
        end

    end
    println("ratio of accepted switches: ", accepted_switches/(accepted_switches+rejected_switches))
    println("number of proposed switches: ", accepted_switches + rejected_switches)
    # return (t_skeleton, x_skeleton, v_skeleton)
    return skel_chain,accepted_switches,rejected_switches
end

function flip!(vel::Vector, i::Integer)
  temp = vel[i]
  vel[i] = -temp
end

function AdaptiveZigZag(∇E::Function, ∂E::Function, Q::Symmetric{Float64,Matrix{Float64}}, T::Real, discr_step::Real, time_adaps::Real;
   x_init::Vector{Float64} = Vector{Float64}(undef,0), v_init::Vector{Int} = Vector{Int}(undef,0), excess_rate::Float64 = 0.0,
   diag_adaptation::Bool = false, return_positions::Bool = false, updating_probs::Function = f(n) = 1/(1+log(n)))
    # ∇E(x) is the gradient of the negative log-likelihood
    # ∂E(i,x) returns the i-th component of ∇E in position x

    dim = size(Q)[1];
    freq_adap = convert(Int64,time_adaps/discr_step)
    if (length(x_init) == 0 || length(v_init) == 0)
        x_init = zeros(dim)
        v_init = rand((-1,1), dim)
    end
    n = 0;
    n_steps = convert(Integer,T/discr_step);
    t = 0.0;
    x = copy(x_init); v = copy(v_init); θ = copy(v_init);
    updateSkeleton = false;
    finished = false;
    x_matrix = Array{Float64,2}(undef,dim,(n_steps + 1))
    x_matrix[:,1] = copy(x)
    skel_chain = skeleton[]
    push!(skel_chain,skeleton(x,v,t))
    index = 1
    t_limit = time_adaps
    iter = 0

    rejected_switches = 0;
    accepted_switches = 0;
    n_refresh = 0;
    a = v.*∇E(x);  # correct also for adaptive
    b = vec([norm(Q[:,i]) for i=1:dim]); # = ||Q M_i||_2
    b = sqrt(dim)*b;  
    Q_norm = opnorm(Q);
    M_norms = ones(dim);

    Δt_proposed_switches = switchingtime.(a,b)
    if (excess_rate == 0.0)
        Δt_excess = Inf
    else
        Δt_excess = -log(rand())/(dim*excess_rate)
    end

    if diag_adaptation   #learn only the diagonal
        M = ones(dim)
        μ_hat = zeros(dim);
        μ_old = zeros(dim);
        Σ_hat = zeros(dim);
        Q_norms = [norm(Q[:,i]) for i=1:dim]
        while (!finished)
            i = argmin(Δt_proposed_switches) # O(d)
            Δt_switch_proposed = Δt_proposed_switches[i]
            Δt = min(Δt_switch_proposed,Δt_excess);

            if (t+Δt) < t_limit   # then simulate from current kernel
                x = x + v * Δt; # O(d)
                # x += v * Δt; # O(d)
                t = t + Δt;
                a = a + b * Δt; # O(d). This remains true
                if (Δt_switch_proposed < Δt_excess)
                    # switch_rate = θ[i] * M[i] * ∂E(i,x)
                    switch_rate = v[i] * ∂E(i,x)
                    proposedSwitchIntensity = a[i]
                    if proposedSwitchIntensity < switch_rate
                        println("ERROR: Switching rate exceeds bound.")
                        println(" simulated rate: ", proposedSwitchIntensity)
                        println(" actual switching rate: ", switch_rate)
                        println(" current time: ", t)
                        println(" number of accepted events: ", accepted_switches)
                        println(" number of rejected events: ", rejected_switches)
                        println("velocity theta = ", θ)
                        println("velocity v = ", v)
                        println("position x = ", x)
                        println("vector a = " , a)
                        println("vector b = " , b)
                        error("Switching rate exceeds bound.")
                    end
                    if rand() * proposedSwitchIntensity <= switch_rate
                        θ[i] = -θ[i]
                        # v[i] = M[i]*θ[i]
                        v[i] = -v[i]   # update velocity of the process
                        a[i] = -switch_rate 
                        updateSkeleton = true
                        accepted_switches += 1
                    else  # reject proposed switch
                        a[i] = switch_rate   # since we computed it, let's use it
                        updateSkeleton = false
                        rejected_switches += 1
                    end
                    # update refreshment time and switching time bound
                    Δt_excess = Δt_excess - Δt_switch_proposed
                    Δt_proposed_switches = Δt_proposed_switches .- Δt_switch_proposed  #the bounds still hold for components that haven't switched
                    # for the line above to work we need bounds b_i that do not depend on θ
                    Δt_proposed_switches[i] = switchingtime(a[i],b[i])
                else  # so we switch due to excess switching rate
                    n_refresh +=1
                    updateSkeleton = true
                    i = rand(1:dim)
                    θ[i] = -θ[i]
                    v[i] = -v[i]   # update velocity of the process
                    a[i] = v[i] * ∂E(i,x)  #have to change this since θ[i] changes

                    # update upcoming event times
                    Δt_proposed_switches = Δt_proposed_switches .- Δt_excess
                    Δt_proposed_switches[i] = switchingtime(a[i],b[i])  # Need to update since old bound doesn't hold
                    Δt_excess = -log(rand())/(dim*excess_rate);
                end
            elseif t + Δt < T # it is time to update the parameters
                #t_new = (n+1)*time_adaps
                Δt = (n+1)*time_adaps - t
                updateSkeleton = true
                x = x + v * Δt; # O(d)
                t = (n+1)*time_adaps;
                x_matrix[:,(iter+2):((n+1)*freq_adap+1)] = discretise(skel_chain[index:end],discr_step, t, dim)
                μ_hat,Σ_hat = update_γ_diag(x_matrix[:,(iter+2):((n+1)*freq_adap+1)],Σ_hat,μ_hat,μ_old,iter)
                μ_old = copy(μ_hat)
                U = rand()
                if t/discr_step > dim && U <= updating_probs(n+1)
                  M = sqrt.(Σ_hat)
                  v = M.*θ
                  b = M.*Q_norms*norm(M)
                end
                # v = M.*θ;
                n += 1
                iter = n*freq_adap
                index = (accepted_switches + n_refresh) + n + 1
                t_limit += time_adaps
                a = v.*∇E(x)
                Δt_proposed_switches = switchingtime.(a,b)
            else   # time to wrap up
                Δt = T - t 
                finished = true
                updateSkeleton = true
                x = x + v * Δt; # O(d)
                t = T;
                x_matrix[:,(iter+2):end] = discretise(skel_chain[index:end],discr_step, t, dim)
            end

            if updateSkeleton
                push!(skel_chain,skeleton(copy(x),copy(v),t))
                updateSkeleton = false
            end
          end

    else   # learn the entire covariance matrix
        M = Matrix{Float64}(I, dim, dim)
        μ_hat = zeros(dim);
        μ_old = zeros(dim);
        Σ_hat = zeros(dim,dim);
        while (!finished)
            i = argmin(Δt_proposed_switches) # O(d)
            Δt_switch_proposed = Δt_proposed_switches[i]
            Δt = min(Δt_switch_proposed,Δt_excess);
            if (t+Δt) < t_limit   # then simulate from current kernel
                x = x + v * Δt; # O(d)
                t = t + Δt;
                a = a + b * Δt; # O(d). This remains true
                if (Δt_switch_proposed < Δt_excess)
                    gradE =∇E(x)
                    switch_rate = θ[i]*dot(M[:,i],gradE)  #need to compute the true switching rate at time t
                    proposedSwitchIntensity = a[i]
                    if proposedSwitchIntensity < switch_rate
                        println("ERROR: Switching rate exceeds bound.")
                        println(" simulated rate: ", proposedSwitchIntensity)
                        println(" actual switching rate: ", switch_rate)
                        error("Switching rate exceeds bound.")
                    end
                    if rand() * proposedSwitchIntensity <= switch_rate
                        # switch i-th component
                        θ[i] = -θ[i]
                        #v -= 2*θ[i]*M[:,i]
                        v = M*θ   # update velocity of the process
                        # a = θ.*transpose(M)*gradE
                        a[i] = -switch_rate   
                        # sharper bound using the current velocity in b
                        #b = Q_norm * norm(v) * M_norms;
                        # but this would then require updating all proposed switching times!
                        updateSkeleton = true
                        accepted_switches += 1
                    else  # reject proposed switch
                        a[i] = switch_rate   # since we computed it, let's use it
                        updateSkeleton = false
                        rejected_switches += 1
                    end
                    # update refreshment time and switching time bound
                    Δt_excess = Δt_excess - Δt_switch_proposed
                    Δt_proposed_switches = Δt_proposed_switches .- Δt_switch_proposed  #the bounds still hold for components that haven't switched...
                    # # for the line above to work we need bounds b_i that do not depend on θ
                    Δt_proposed_switches[i] = switchingtime(a[i],b[i])
                else  # so we switch due to excess switching rate
                    n_refresh +=1
                    updateSkeleton = true
                    i = rand(1:dim)
                    θ[i] = -θ[i]
                    v = M*θ   # update velocity of the process
                    a[i] = θ[i] * dot(M[:,i],∇E(x))  #have to change this since θ[i] changes

                    # update upcoming event times
                    Δt_proposed_switches = Δt_proposed_switches .- Δt_excess
                    Δt_proposed_switches[i] = switchingtime(a[i],b[i])  # Need to update since old bound doesn't hold
                    Δt_excess = -log(rand())/(dim*excess_rate);
                end
            elseif t + Δt < T # it is time to update the parameters
                #t_new = (n+1)*time_adaps
                Δt = (n+1)*time_adaps - t
                updateSkeleton = true
                x = x + v * Δt; # O(d)
                t = (n+1)*time_adaps;
                x_matrix[:,(iter+2):((n+1)*freq_adap+1)] = discretise(skel_chain[index:end],discr_step, t, dim)
                μ_hat,Σ_hat = update_γ(x_matrix[:,(iter+2):((n+1)*freq_adap+1)],Σ_hat,μ_hat,μ_old,iter)
                μ_old = copy(μ_hat)
                U = rand()
                if t/discr_step > dim && U <= updating_probs(n+1)
                  M = convert(Array{Float64,2},sqrt(Σ_hat))
                  v = M*θ;
                end
                # v = M*θ;
                n += 1
                iter = n*freq_adap
                index = (accepted_switches + n_refresh) + n + 1
                t_limit += time_adaps
                a = θ.*(transpose(M)*∇E(x))  #no transpose since M symmetric
                # M_norms = [norm(M[:,i]) for i=1:dim]
                # b = Q_norm * norm(v) * M_norms;
                b_common = sqrt(dim)*opnorm(M,2)
                b = [norm(Q*M[:,i]) for i=1:dim]
                b = b_common * b
                Δt_proposed_switches = switchingtime.(a,b)
            else   # time to wrap up
                Δt = T - t 
                finished = true
                updateSkeleton = true
                x = x + v * Δt; # O(d)
                t = T;
                x_matrix[:,(iter+2):end] = discretise(skel_chain[index:end],discr_step, t, dim)
            end

            if updateSkeleton
                push!(skel_chain,skeleton(copy(x),copy(v),t))
                updateSkeleton = false
            end
          end
    end
    println("ratio of accepted switches: ", accepted_switches/(accepted_switches+rejected_switches))
    println("number of proposed switches: ", accepted_switches + rejected_switches)
    if return_positions
      return (x_matrix,skel_chain)
    else
      return skel_chain,accepted_switches,rejected_switches
    end

end


function AdaptiveZigZagSubsampling(∇E::Function, ∂E::Function, Q::Symmetric{Float64,Matrix{Float64}}, N::Int, T::Real, discr_step::Real, time_adaps::Real;
   x_ref::Vector{Float64} = Vector{Float64}(undef,0), ∇E_ref::Vector{Float64} = Vector{Float64}(undef,0), x_init::Vector{Float64} = Vector{Float64}(undef,0),
   v_init::Vector{Int} = Vector{Int}(undef,0), excess_rate::Float64 = 0.0, diag_adaptation::Bool = false, return_positions::Bool = false, updating_probs::Function = f(n) = 1/(1+log(n)))
# Need input both ∇E and ∂E. The former is a function that has two inputs j and x, indicating
# respectively the j-the data point and the position. The latter is a function with three inputs
# i (i-th component), j (j-th data point), x (position). ∇E is used for the full adaptive ZZS
# while ∂E is used when diagonal adaptation is selected


    if (length(x_ref) == 0 || length(∇E_ref) == 0)
        controlvariates = false;
        error("ZigZagSubampling without control variates currently not supported")
    else
        controlvariates = true;
    end

    dim = size(Q)[1]
    if length(x_init) == 0
        if controlvariates
            x_init = x_ref
        else
            x_init = zeros(dim)
        end
    end
    if length(v_init) == 0
        v_init = rand((-1,1), dim)
    end

    freq_adap = convert(Int64,time_adaps/discr_step)
    n_steps = convert(Integer,T/discr_step);

    t = 0.0;
    n = 0;
    x = x_init; v = copy(v_init); θ = copy(v_init);
    x_matrix = Array{Float64,2}(undef,dim,(n_steps + 1))
    x_matrix[:,1] = x
    updateSkeleton = false;
    finished = false;
    skel_chain = skeleton[]
    push!(skel_chain,skeleton(x,v,t))
    rejected_switches = 0;
    accepted_switches = 0;
    n_refresh = 0;
    index = 1
    t_limit = time_adaps
    iter = 0
    distance_to_ref = 0;               # we start in the reference point
    a = θ.*∇E_ref                      # we start with M = identity matrix and in the reference point
    C = vec(sqrt.(sum(Q.^2, dims=2)))  # use bound on the Hessian to bound the Lipschitz constants.
    b = C * sqrt(dim);                       # assuming Euclidean norm and M=I at the beginning
    Δt_proposed_switches = switchingtime.(a,b)

    # define quantities that do not change if M remains the same
    absM_prod_C = C;   # assuming for the moment M=I
    M_prod_∇E_ref = ∇E_ref;

    if (excess_rate == 0.0)
        Δt_excess = Inf
    else
        Δt_excess = -log(rand())/(dim*excess_rate)
    end

    if diag_adaptation
        M = ones(dim)
        μ_hat = zeros(dim);
        μ_old = zeros(dim);
        Σ_hat = zeros(dim);
        while (!finished)
            i = argmin(Δt_proposed_switches) # O(d)
            Δt_switch_proposed = Δt_proposed_switches[i]
            Δt = min(Δt_switch_proposed,Δt_excess); 
            if (t+Δt) < t_limit   # then simulate from current kernel
                x = x + v * Δt; # O(d)
                t = t + Δt;
                a = a + b * Δt; # O(d). This remains true
                if (Δt_switch_proposed < Δt_excess)
                    # switch_rate = θ[i]*M_prod_∇E_ref[i] + absM_prod_C[i]*norm(x-x_ref)  #need to compute the true switching rate at time t
                    j = rand(1:N)
                    switch_rate = θ[i] * M[i] * (∇E_ref[i] + ∂E(i,j,x) - ∂E(i,j,x_ref)) 
                    proposedSwitchIntensity = a[i]
                    if proposedSwitchIntensity < switch_rate
                        println("ERROR: Switching rate exceeds bound.")
                        println(" simulated rate: ", proposedSwitchIntensity)
                        println(" actual switching rate: ", switch_rate)
                        error("Switching rate exceeds bound.")
                    end
                    if rand() * proposedSwitchIntensity <= switch_rate
                        # switch i-th component
                        θ[i] = -θ[i]
                        v = M.*θ         # update velocity of the process
                        a[i] = θ[i]*M_prod_∇E_ref[i] + absM_prod_C[i]*norm(x-x_ref) 
                        updateSkeleton = true
                        accepted_switches += 1
                    else  # reject proposed switch
                        # a[i] = switch_rate   # no need to change a[i]
                        updateSkeleton = false
                        rejected_switches += 1
                    end
                    # update refreshment time and switching time bound
                    Δt_excess = Δt_excess - Δt_switch_proposed
                    Δt_proposed_switches = Δt_proposed_switches .- Δt_switch_proposed  #the bounds still hold for components that haven't switched...
                    # for the line above to work we need bounds b_i that do not depend on θ
                    Δt_proposed_switches[i] = switchingtime(a[i],b[i])
                else  # so we switch due to excess switching rate
                    n_refresh +=1
                    updateSkeleton = true
                    i = rand(1:dim)
                    θ[i] = -θ[i]
                    v = M.*θ   # update velocity of the process
                    a[i] = θ[i]*M_prod_∇E_ref[i] + absM_prod_C[i]*norm(x-x_ref)  #have to change this since θ[i] and x have changed

                    # update upcoming event times
                    Δt_proposed_switches = Δt_proposed_switches .- Δt_excess
                    Δt_proposed_switches[i] = switchingtime(a[i],b[i])  # Need to update since old bound doesn't hold
                    Δt_excess = -log(rand())/(dim*excess_rate);
                end
            elseif t + Δt < T # it is time to update the parameters
                #t_new = (n+1)*time_adaps
                Δt = (n+1)*time_adaps - t
                updateSkeleton = true
                x = x + v * Δt; # O(d)
                t = (n+1)*time_adaps;
                x_matrix[:,(iter+2):((n+1)*freq_adap+1)] = discretise(skel_chain[index:end],discr_step, t, dim)
                μ_hat,Σ_hat = update_γ_diag(x_matrix[:,(iter+2):((n+1)*freq_adap+1)],Σ_hat,μ_hat,μ_old,iter)
                μ_old = copy(μ_hat)
                U=rand()
                if t/discr_step > dim && U<=updating_probs(n+1)
                  M = sqrt.(Σ_hat)
                  v = M.*θ;
                end
                n += 1
                iter = n*freq_adap
                index = (accepted_switches + n_refresh) + n + 1
                t_limit += time_adaps
                # update bounds and simulate next switching times
                absM_prod_C = (abs.(M)).*C;
                M_prod_∇E_ref = M.*∇E_ref;
                a = θ.*M_prod_∇E_ref + absM_prod_C*norm(x-x_ref)
                b = sqrt(dim)*norm(M)*absM_prod_C
                Δt_proposed_switches = switchingtime.(a,b)
            else   # time to wrap up
                Δt = T - t # correct to get right to the end of the time horizon
                finished = true
                updateSkeleton = true
                x = x + v * Δt; # O(d)
                t = T;
                x_matrix[:,(iter+2):end] = discretise(skel_chain[index:end],discr_step, t, dim)
            end

            if updateSkeleton
                push!(skel_chain,skeleton(x,v,t))
                updateSkeleton = false
            end
          end
      else  # learn the entire covariance matrix
          M = Matrix{Float64}(I, dim, dim)
          μ_hat = zeros(dim);
          μ_old = zeros(dim);
          Σ_hat = zeros(dim,dim);
          while (!finished)
              i = argmin(Δt_proposed_switches) # O(d)
              Δt_switch_proposed = Δt_proposed_switches[i]
              Δt = min(Δt_switch_proposed,Δt_excess);  
              if (t+Δt) < t_limit   # then simulate from current kernel
                  x = x + v * Δt; # O(d)
                  t = t + Δt;
                  a = a + b * Δt; # O(d). This remains true
                  if (Δt_switch_proposed < Δt_excess)
                      j = rand(1:N)
                      switch_rate = θ[i] * dot(M[:,i], ∇E_ref + ∇E(j,x) - ∇E(j,x_ref)) 
                      proposedSwitchIntensity = a[i]
                      if proposedSwitchIntensity < switch_rate
                          println("ERROR: Switching rate exceeds bound.")
                          println(" simulated rate: ", proposedSwitchIntensity)
                          println(" actual switching rate: ", switch_rate)
                          println(" CURRENT TIME: ", t)
                          println(" accepted switches:", accepted_switches)
                          error("Switching rate exceeds bound.")
                      end
                      if rand() * proposedSwitchIntensity <= switch_rate
                          # switch i-th component
                          θ[i] = -θ[i]
                          v = M*θ         # update velocity of the process
                          a[i] = θ[i]*M_prod_∇E_ref[i] + absM_prod_C[i]*norm(x-x_ref)   # change sign of the first term of a[i]
                          updateSkeleton = true
                          accepted_switches += 1
                      else  # reject proposed switch
                          # a[i] = switch_rate   # no need to change a[i]
                          updateSkeleton = false
                          rejected_switches += 1
                      end
                      # update refreshment time and switching time bound
                      Δt_excess = Δt_excess - Δt_switch_proposed
                      Δt_proposed_switches = Δt_proposed_switches .- Δt_switch_proposed  #the bounds still hold for components that haven't switched...
                      # for the line above to work we need bounds b_i that do not depend on θ
                      Δt_proposed_switches[i] = switchingtime(a[i],b[i])
                  else  # so we switch due to excess switching rate
                      n_refresh +=1
                      updateSkeleton = true
                      i = rand(1:dim)
                      θ[i] = -θ[i]
                      v = M*θ   # update velocity of the process
                      a[i] = θ[i]*M_prod_∇E_ref[i] + absM_prod_C[i]*norm(x-x_ref)  #have to change this since θ[i] and x have changed

                      # update upcoming event times
                      Δt_proposed_switches = Δt_proposed_switches .- Δt_excess
                      Δt_proposed_switches[i] = switchingtime(a[i],b[i])  # Need to update since old bound doesn't hold
                      Δt_excess = -log(rand())/(dim*excess_rate);
                  end
              elseif t + Δt < T # it is time to update the parameters
                  #t_new = (n+1)*time_adaps
                  Δt = (n+1)*time_adaps - t
                  updateSkeleton = true
                  x = x + v * Δt; # O(d)
                  t = (n+1)*time_adaps;
                  x_matrix[:,(iter+2):((n+1)*freq_adap+1)] = discretise(skel_chain[index:end],discr_step, t, dim)
                  μ_hat,Σ_hat = update_γ(x_matrix[:,(iter+2):((n+1)*freq_adap+1)],Σ_hat,μ_hat,μ_old,iter)
                  μ_old = copy(μ_hat)
                  U = rand()
                  if t/discr_step > dim && U <= updating_probs(n+1)
                    M = convert(Array{Float64,2},sqrt(Σ_hat))
                    v = M*θ;
                  end
                  n += 1
                  iter = n*freq_adap
                  index = (accepted_switches + n_refresh) + n + 1
                  t_limit += time_adaps
                  # update bounds and simulate next switching times
                  absM_prod_C = abs.(M)*C;
                  M_prod_∇E_ref = M*∇E_ref;
                  a = θ.*M_prod_∇E_ref + absM_prod_C*norm(x-x_ref)
                  b = sqrt(dim)*norm(M)*absM_prod_C
                  Δt_proposed_switches = switchingtime.(a,b)
              else   # time to wrap up
                  Δt = T - t # correct to get right to the end of the time horizon
                  finished = true
                  updateSkeleton = true
                  x = x + v * Δt; # O(d)
                  t = T;
                  x_matrix[:,(iter+2):end] = discretise(skel_chain[index:end],discr_step, t, dim)
              end

              if updateSkeleton
                  push!(skel_chain,skeleton(x,v,t))
                  updateSkeleton = false
              end
        end
    end
    println("ratio of accepted switches: ", accepted_switches/(accepted_switches+rejected_switches))
    println("number of proposed switches: ", accepted_switches + rejected_switches)
    if return_positions
      return (x_matrix,skel_chain)
    else
      return skel_chain,accepted_switches,rejected_switches
    end

end

function reflect(gradient::Vector{Float64}, v::Vector{Float64})

    return v - 2 * (transpose(gradient) * v / dot(gradient,gradient)) * gradient

end

tolerance = 1e-7 # for comparing switching rate and bound

function BPS(∇E::Function, Q::Symmetric{Float64,Matrix{Float64}}, T::Real; x_init::Vector{Float64} = Vector{Float64}(undef,0), v_init::Vector{Float64} = Vector{Float64}(undef,0), refresh_rate::Float64 = 1.0)
    # g_E! is the gradient of the energy function E
    # Q is a symmetric matrix such that Q - ∇^2 E(x) is positive semidefinite

    dim = size(Q)[1]
    if (length(x_init) == 0 || length(v_init) == 0)
        x_init = zeros(dim)
        v_init = randn(dim)
    end

    t = 0.0;
    x = x_init; v = v_init;
    updateSkeleton = false;
    finished = false;

    skel_chain = skeleton[]
    push!(skel_chain,skeleton(x,v,t))

    rejected_switches = 0;
    accepted_switches = 0;
    gradient = ∇E(x);
    a = transpose(v) * gradient;
    b = transpose(v) * Q * v;

    Δt_switch_proposed = switchingtime(a,b)
    if refresh_rate <= 0.0
        Δt_refresh = Inf
    else
        Δt_refresh = -log(rand())/refresh_rate
    end

    while (!finished)
        Δt = min(Δt_switch_proposed,Δt_refresh);
        if t + Δt > T
            Δt = T - t
            finished = true
            updateSkeleton = true
        end
        x = x + v * Δt; # O(d)
        t = t + Δt;
        a = a + b * Δt; # O(d)
        gradient = ∇E(x)

        if (!finished && Δt_switch_proposed < Δt_refresh)
            switch_rate = transpose(v) * gradient
            proposedSwitchIntensity = a
            if proposedSwitchIntensity < switch_rate - tolerance
                println("ERROR: Switching rate exceeds bound.")
                println(" simulated rate: ", proposedSwitchIntensity)
                println(" actual switching rate: ", switch_rate)
                error("Switching rate exceeds bound.")
            end
            if rand() * proposedSwitchIntensity <= switch_rate
                # switch i-th component
                v = reflect(gradient,v)
                a = -switch_rate
                b = transpose(v) * Q * v
                updateSkeleton = true
                accepted_switches += 1
            else
                a = switch_rate
                updateSkeleton = false
                rejected_switches += 1
            end
            # update time to refresh
            Δt_refresh = Δt_refresh - Δt_switch_proposed
        elseif !finished
            # so we refresh
            updateSkeleton = true
            v = randn(dim)
            a = transpose(v) * gradient
            b = transpose(v) * Q * v

            # update time to refresh
            Δt_refresh = -log(rand())/refresh_rate;
        end

        if updateSkeleton
            # push!(x_skeleton, x)
            # push!(v_skeleton, v)
            # push!(t_skeleton, t)
            push!(skel_chain,skeleton(x,v,t))
            updateSkeleton = false
        end
        Δt_switch_proposed = switchingtime(a,b)
    end
    println("ratio of accepted switches: ", accepted_switches/(accepted_switches+rejected_switches))
    println("number of proposed switches: ", accepted_switches + rejected_switches)
    return skel_chain,accepted_switches,rejected_switches
end


function BPS_subsampling(∇E_ss::Function, hessian_bound_ss::Float64, n_observations::Int, T::Float64, x_ref::Vector{Float64} = Vector{Float64}(undef,0), ∇E_ref::Vector{Float64} = Vector{Float64}(undef,0); x_init::Vector{Float64} = Vector{Float64}(undef,0), v_init::Vector{Int} = Vector{Int}(undef,0), refresh_rate::Float64 = 1.0)
# BPS with subsampling and control variates
# grad_E_ss(l,x) gives the ∇ E^l(x), where 1/N ∑_{l=1}^N E^l(x) = E(x), the full potential function
# Q is a symmetric matrix such that - Q ⪯ ∇^2 E^l(x) ⪯ Q for all l,x


    if (length(x_ref) == 0 || length(∇E_ref) == 0)
        controlvariates = false;
        error("BPS_Subampling without control variates currently not supported")
    else
        controlvariates = true;
    end

    dim = length(x_ref)
    if length(x_init) == 0
        if controlvariates
            x_init = x_ref
        else
            x_init = zeros(dim)
        end
    end
    if length(v_init) == 0
        v_init = randn(dim)
    end

    t = 0.0;
    x = x_init; v = v_init;
    updateSkeleton = false;
    finished = false;
    skel_chain = skeleton[]
    push!(skel_chain, skeleton(x,v,t))

    rejected_switches = 0;
    accepted_switches = 0;
    sqnorm_v = dot(v,v)
    b = hessian_bound_ss * sqnorm_v
    a = dot(v, ∇E_ref) + hessian_bound_ss * sqrt(sqnorm_v) * norm(x-x_ref)
    Δt_switch_proposed = switchingtime(a,b)

    if (refresh_rate == 0.0)
        Δt_refresh = Inf
    else
        Δt_refresh = -log(rand())/refresh_rate
    end

    while (!finished)
        Δt = min(Δt_switch_proposed,Δt_refresh);
        if t + Δt > T
            Δt = T - t
            finished = true
            updateSkeleton = true
        end
        x = x + v * Δt; # O(d)
        t = t + Δt;
        a = a + b * Δt; # O(d)

        if (!finished && Δt_switch_proposed < Δt_refresh)
            j = rand(1:n_observations)
            ∇E_est = ∇E_ref + ∇E_ss(j,x) - ∇E_ss(j,x_ref)
            switch_rate = dot(v, ∇E_est)
            proposal_intensity = a
            if proposal_intensity < switch_rate
                println("ERROR: Switching rate exceeds bound.")
                println(" simulated rate: ", proposal_intensity)
                println(" actual switching rate: ", switch_rate)
                error("Switching rate exceeds bound.")
            end
            if rand() * proposal_intensity <= switch_rate
                # reflect
                v = v - 2 * dot(v, ∇E_est)/dot(∇E_est,∇E_est)*∇E_est
                updateSkeleton = true
                accepted_switches += 1
            else
                updateSkeleton = false
                rejected_switches += 1
            end
            # update refreshment time and switching time bound
            Δt_refresh = Δt_refresh - Δt_switch_proposed
        elseif !finished
            # so we refresh
            updateSkeleton = true
            v = randn(dim)
            # update upcoming event times
            Δt_refresh = -log(rand())/refresh_rate;
        end

        if updateSkeleton
            push!(skel_chain, skeleton(x,v,t))
            updateSkeleton = false
            b = hessian_bound_ss * dot(v,v) # norm v changes so we update this
        end

        a = dot(v, ∇E_ref) + sqrt(b) * sqrt(hessian_bound_ss) * sqrt(dot(x-x_ref,x-x_ref))
        Δt_switch_proposed = switchingtime(a,b)

    end
    println("ratio of accepted switches: ", accepted_switches/(accepted_switches+rejected_switches))
    println("number of proposed switches: ", accepted_switches + rejected_switches)
    # return (t_skeleton, x_skeleton, v_skeleton)
    return skel_chain,accepted_switches,rejected_switches
end



function AdaptiveBPS(∇E::Function, Q::Symmetric{Float64,Matrix{Float64}}, T::Real, discr_step::Real, time_adaps::Real;
          x_init::Vector{Float64} = Vector{Float64}(undef,0), v_init::Vector{Float64} = Vector{Float64}(undef,0),
          refresh_rate::Float64 = 1.0, λBool::Bool = false, diag_adaptation::Bool = false, preserve_nr_events::Bool = false,
          return_positions::Bool = false, only_refresh::Bool = false, updating_probs::Function = f(n) = 1/(1+log(n)))
    # ∇E is the gradient of the energy function E
    # Q is a symmetric matrix such that Q - ∇^2 E(x) is positive semidefinite
    freq_adap = convert(Int64,time_adaps/discr_step) #assuming this gives a natural number
    dim = size(Q)[1]
    if (length(x_init) == 0 || length(v_init) == 0)
        x_init = zeros(dim)
        v_init = randn(dim)
    end

    t = 0.0;
    n = 0; n_steps = convert(Integer,T/discr_step);
    x = x_init; v = copy(v_init); θ = copy(v_init);
    x_matrix = Array{Float64,2}(undef,dim,(n_steps + 1))
    x_matrix[:,1] = x
    skel_chain = skeleton[]
    push!(skel_chain,skeleton(x,v,t))
    index = 1
    t_limit = time_adaps
    iter = 0
    old_bounces = 0; old_events = 0;
    skipped_adaptations = 1
    updateSkeleton = false;
    finished = false;
    rejected_switches = 0;
    accepted_switches = 0;
    n_refresh = 0;
    gradient = ∇E(x);
    a = transpose(v) * gradient;
    b = transpose(v) * Q * v;

    Δt_switch_proposed = switchingtime(a,b)
    if refresh_rate <= 0.0
        Δt_refresh = Inf
    else
        Δt_refresh = -log(rand())/refresh_rate
    end
    if !only_refresh
      if diag_adaptation
          M = ones(dim);
          μ_hat = zeros(dim);
          μ_old = zeros(dim);
          Σ_hat = zeros(dim);

          while (!finished)
              Δt = min(Δt_switch_proposed,Δt_refresh);
              if (t+Δt) < t_limit   # then simulate from current kernel
                  x = x + v * Δt; # O(d)
                  t = t + Δt;
                  # a = a + b * Δt; # O(d).
                  gradient = ∇E(x)
                  if (Δt_switch_proposed < Δt_refresh)
                      a = a + b * Δt; # O(d).
                      switch_rate = transpose(v) * gradient
                      proposedSwitchIntensity = a
                      if proposedSwitchIntensity < switch_rate - tolerance
                          println("ERROR: Switching rate exceeds bound.")
                          println(" simulated rate: ", proposedSwitchIntensity)
                          println(" actual switching rate: ", switch_rate)
                          error("Switching rate exceeds bound.")
                      end
                      if rand() * proposedSwitchIntensity <= switch_rate
                          transformed_gradient = M.*gradient
                          θ = reflect(transformed_gradient,θ)
                          v = M.*θ
                          a = -switch_rate
                          b = transpose(v) * Q * v
                          updateSkeleton = true
                          accepted_switches += 1
                      else  # reject proposed switch
                          a = switch_rate   # since we computed it, let's use it, no  need to update b as Θ is not reflected and it does not depend on x
                          updateSkeleton = false
                          rejected_switches += 1
                      end
                      Δt_refresh = Δt_refresh - Δt_switch_proposed
                      Δt_switch_proposed = switchingtime(a,b)
                  else  # refreshment
                    n_refresh +=1
                    updateSkeleton = true
                    θ = randn(dim)
                    v = M.*θ
                    a = transpose(v) * gradient
                    b = transpose(v) * Q * v

                    # update time to refresh
                    Δt_refresh = -log(rand())/refresh_rate;
                    Δt_switch_proposed = switchingtime(a,b)
                  end
              elseif t + Δt < T     # then it is time to update the parameters
                  Δt = (n+1)*time_adaps - t
                  updateSkeleton = true
                  x = x + v * Δt; # O(d)
                  t = (n+1)*time_adaps;
                  x_matrix[:,(iter+2):((n+1)*freq_adap+1)] = discretise(skel_chain[index:end],discr_step, t, dim)
                  μ_hat,Σ_hat = update_γ_diag(x_matrix[:,(iter+2):((n+1)*freq_adap+1)],Σ_hat,μ_hat,μ_old,iter)
                  μ_old = copy(μ_hat)
                  U = rand()
                  if (t/discr_step > dim) && (U <= updating_probs(n+1))
                    M = sqrt.(Σ_hat)
                    v = M.*θ;
                  end
                  # v = M.*θ;
                  n += 1
                  iter = n*freq_adap
                  index = (accepted_switches + n_refresh) + n + 1
                  t_limit += time_adaps
                  gradient = ∇E(x)
                  a = transpose(v) * gradient
                  b = transpose(v) * Q * v
                  Δt_switch_proposed = switchingtime(a,b)
                  if λBool && (U <= updating_probs(n))
                    # new_events  = accepted_switches+n_refresh-old_events
                    new_bounces = accepted_switches-old_bounces
                    reflection_rate = new_bounces/(skipped_adaptations * time_adaps)
                    refresh_rate = 0.7812*reflection_rate/0.2188
                    # if (new_bounces)/(new_events) > 1 - 0.7812
                    #   # refresh_rate = min(10,refresh_rate + 1/log(t))
                    #   refresh_rate = min(10,refresh_rate + 1/(n^2))
                    # else
                    #   # refresh_rate = max(0.01, refresh_rate - 1/log(t))
                    #   refresh_rate = max(0.01, refresh_rate - 1/(n^2))
                    # end
                    old_bounces = accepted_switches
                    # old_events  = accepted_switches+n_refresh
                    skipped_adaptations = 1
                  elseif λBool && (U > updating_probs(n))
                    skipped_adaptations += 1
                  end
                  Δt_refresh = -log(rand())/refresh_rate;
              else   # time to wrap up
                  Δt = T - t # correct to get right to the end of the time horizon
                  finished = true
                  updateSkeleton = true
                  x = x + v * Δt; # O(d)
                  t = T;
                  x_matrix[:,(iter+2):end] = discretise(skel_chain[index:end],discr_step, t, dim)
                  println("Empirical covariance is: ", Σ_hat)
              end
              if updateSkeleton
                  push!(skel_chain,skeleton(x,v,t))
                  updateSkeleton = false
              end
          end

        else  # Learn the full covariance matrix

            M = Matrix{Float64}(I, dim, dim)
            μ_hat = zeros(dim);
            μ_old = zeros(dim);
            Σ_hat = zeros(dim,dim);

            while (!finished)
                Δt = min(Δt_switch_proposed,Δt_refresh);
                if (t+Δt) < t_limit   # then simulate from current kernel
                    x = x + v * Δt; # O(d)
                    t = t + Δt;
                    # a = a + b * Δt; # O(d).
                    gradient = ∇E(x)
                    if (Δt_switch_proposed < Δt_refresh)
                        a = a + b * Δt; # O(d).
                        switch_rate = transpose(v) * gradient
                        proposedSwitchIntensity = a
                        if proposedSwitchIntensity < switch_rate - tolerance
                            println("ERROR: Switching rate exceeds bound.")
                            println(" simulated rate: ", proposedSwitchIntensity)
                            println(" actual switching rate: ", switch_rate)
                            error("Switching rate exceeds bound.")
                        end
                        if rand() * proposedSwitchIntensity <= switch_rate
                            transformed_gradient = M*gradient   # assuming M is symmetric
                            θ = reflect(transformed_gradient,θ)
                            v = M*θ
                            a = -switch_rate
                            b = transpose(v) * Q * v   # It is also possible by Cauchy's inequality to choose b so that it doesn't need to be updated every time
                            updateSkeleton = true
                            accepted_switches += 1
                        else  # reject proposed switch
                            a = switch_rate   # since we computed it, let's use it, no  need to update b as Θ is not reflected and it does not depend on x
                            updateSkeleton = false
                            rejected_switches += 1
                        end
                        Δt_refresh = Δt_refresh - Δt_switch_proposed
                        Δt_switch_proposed = switchingtime(a,b)
                    else  # refreshment
                      updateSkeleton = true
                      θ = randn(dim)
                      v = M*θ
                      a = transpose(v) * gradient
                      b = transpose(v) * Q * v
                      n_refresh +=1
                      # update time to refresh
                      Δt_refresh = -log(rand())/refresh_rate;
                      Δt_switch_proposed = switchingtime(a,b)
                    end
                elseif t + Δt < T     # then it is time to update the parameters
                    Δt = (n+1)*time_adaps - t
                    updateSkeleton = true
                    x = x + v * Δt; # O(d)
                    t = (n+1)*time_adaps;
                    x_matrix[:,(iter+2):((n+1)*freq_adap+1)] = discretise(skel_chain[index:end],discr_step, t, dim)
                    μ_hat,Σ_hat = update_γ(x_matrix[:,(iter+2):((n+1)*freq_adap+1)],Σ_hat,μ_hat,μ_old,iter)
                    μ_old = copy(μ_hat)
                    U = rand()
                    if (t/discr_step > dim) && (U <= updating_probs(n+1))
                      #Σ_hat = Symmetric(Σ_hat)
                      if preserve_nr_events
                        M = sqrt(dim/tr(Σ_hat)) * Symmetric(sqrt(Σ_hat))
                      else
                        # M = convert(Array{Float64,2},sqrt(Σ_hat))
                        M = Symmetric(sqrt(Σ_hat))
                        # M = convert(Symmetric{Float64,Array{Float64,2},sqrt(Σ_hat))
                      end
                      v = M*θ;
                    end
                    # v = M*θ;
                    n += 1
                    iter = n*freq_adap
                    index = (accepted_switches + n_refresh) + n + 1
                    t_limit += time_adaps
                    gradient = ∇E(x)
                    a = transpose(v) * gradient
                    b = transpose(v) * Q * v
                    Δt_switch_proposed = switchingtime(a,b)
                    if λBool && (U <= updating_probs(n))
                      new_bounces = accepted_switches-old_bounces
                      reflection_rate = new_bounces/(skipped_adaptations * time_adaps)
                      refresh_rate = 0.7812*reflection_rate/0.2188
                      old_bounces = accepted_switches
                      skipped_adaptations = 1
                    elseif λBool && (U > updating_probs(n))
                      skipped_adaptations += 1
                    end
                    # if λBool && (U <= updating_probs(n+1))
                    #   new_events  = accepted_switches+n_refresh-old_events
                    #   new_bounces = accepted_switches-old_bounces
                    #   if (new_bounces)/(new_events) > 1-0.7812
                    #     # refresh_rate = min(10,refresh_rate + 1/log(t))
                    #     refresh_rate = min(10,refresh_rate + 1/(n^2))
                    #   else
                    #     # refresh_rate = max(0.01, refresh_rate - 1/log(t))
                    #     refresh_rate = max(0.01, refresh_rate - 1/(n^2))
                    #   end
                    #   old_bounces = accepted_switches
                    #   old_events  = accepted_switches+n_refresh
                    # end
                    Δt_refresh = -log(rand())/refresh_rate;
                else   # time to wrap up
                    Δt = T - t # correct to get right to the end of the time horizon
                    finished = true
                    updateSkeleton = true
                    x = x + v * Δt; # O(d)
                    t = T;
                    x_matrix[:,(iter+2):end] = discretise(skel_chain[index:end],discr_step, t, dim)
                    println("Empirical covariance is: ", Σ_hat)
                end
                if updateSkeleton
                    push!(skel_chain,skeleton(x,v,t))
                    updateSkeleton = false
                end
            end
          end
          else # then want to adapt only refreshment rate
            while (!finished)
                Δt = min(Δt_switch_proposed,Δt_refresh);
                if (t+Δt) < t_limit   # then simulate from current kernel
                    x = x + v * Δt; # O(d)
                    t = t + Δt;
                    # a = a + b * Δt; # O(d).
                    gradient = ∇E(x)
                    if (Δt_switch_proposed < Δt_refresh)
                        a = a + b * Δt; # O(d).
                        switch_rate = transpose(v) * gradient
                        proposedSwitchIntensity = a
                        if proposedSwitchIntensity < switch_rate - tolerance
                            println("ERROR: Switching rate exceeds bound.")
                            println(" simulated rate: ", proposedSwitchIntensity)
                            println(" actual switching rate: ", switch_rate)
                            error("Switching rate exceeds bound.")
                        end
                        if rand() * proposedSwitchIntensity <= switch_rate
                            v = reflect(gradient,v)
                            a = -switch_rate
                            b = transpose(v) * Q * v   # It is also possible by Cauchy's inequality to choose b so that it doesn't need to be updated every time
                            updateSkeleton = true
                            accepted_switches += 1
                        else  # reject proposed switch
                            a = switch_rate   # since we computed it, let's use it, no  need to update b as Θ is not reflected and it does not depend on x
                            updateSkeleton = false
                            rejected_switches += 1
                        end
                        Δt_refresh = Δt_refresh - Δt_switch_proposed
                        Δt_switch_proposed = switchingtime(a,b)
                    else  # refreshment
                      updateSkeleton = true
                      v = randn(dim)
                      a = transpose(v) * gradient
                      b = transpose(v) * Q * v
                      n_refresh +=1
                      # update time to refresh
                      Δt_refresh = -log(rand())/refresh_rate;
                      Δt_switch_proposed = switchingtime(a,b)
                    end
                elseif t + Δt < T     # then it is time to update the refreshment rate
                    Δt = (n+1)*time_adaps - t
                    updateSkeleton = true
                    x = x + v * Δt; # O(d)
                    t = (n+1)*time_adaps;
                    n += 1
                    t_limit += time_adaps
                    gradient = ∇E(x)
                    a = transpose(v) * gradient
                    b = transpose(v) * Q * v
                    Δt_switch_proposed = switchingtime(a,b)
                    U = rand()
                    if (U <= updating_probs(n))
                      new_bounces = accepted_switches-old_bounces
                      reflection_rate = new_bounces/(skipped_adaptations * time_adaps)
                      # println("Reflection rate is ",reflection_rate)
                      refresh_rate = 0.7812*reflection_rate/0.2188
                      old_bounces = accepted_switches
                      skipped_adaptations = 1
                    else
                      skipped_adaptations += 1
                    end
                    # if (U <= updating_probs(n+1))
                    #   new_events  = accepted_switches+n_refresh-old_events
                    #   new_bounces = accepted_switches-old_bounces
                    #   if (new_bounces)/(new_events) > 1-0.7812
                    #     # refresh_rate = min(10,refresh_rate + 1/log(t))
                    #     refresh_rate = min(10,refresh_rate + 1/(n^(3/2)))
                    #   else
                    #     # refresh_rate = max(0.01, refresh_rate - 1/log(t))
                    #     refresh_rate = max(0.01, refresh_rate - 1/(n^(3/2)))
                    #   end
                    #   old_bounces = accepted_switches
                    #   old_events  = accepted_switches+n_refresh
                    # end
                    Δt_refresh = -log(rand())/refresh_rate;
                else   # time to wrap up
                    Δt = T - t # correct to get right to the end of the time horizon
                    finished = true
                    updateSkeleton = true
                    x = x + v * Δt; # O(d)
                    t = T;
                    x_matrix[:,(iter+2):end] = discretise(skel_chain[index:end],discr_step, t, dim)
                end
                if updateSkeleton
                    push!(skel_chain,skeleton(x,v,t))
                    updateSkeleton = false
                end

          end
    end
    println("ratio of accepted switches: ", accepted_switches/(accepted_switches+rejected_switches))
    println("number of proposed switches: ", accepted_switches + rejected_switches)
    println("final refreshment rate: ", refresh_rate)
    if return_positions
      return (x_matrix,skel_chain)
    else
      return skel_chain,accepted_switches,rejected_switches
    end

end




function AdaptiveBPS_subsampling(∇E_ss::Function, hessian_bound_ss::Float64, n_observations::Int, T::Real, discr_step::Real,
  time_adaps::Real, x_ref::Vector{Float64} = Vector{Float64}(undef,0), ∇E_ref::Vector{Float64} = Vector{Float64}(undef,0);
  x_init::Vector{Float64} = Vector{Float64}(undef,0), v_init::Vector{Int} = Vector{Int}(undef,0), refresh_rate::Float64 = 1.0,
  adapt_refresh::Bool = false, diag_adaptation::Bool = false, preserve_nr_events::Bool = false,
  return_positions::Bool = false, only_refresh::Bool = false, updating_probs::Function = f(n) = 1/(1+log(n)))
# BPS with subsampling and control variates
# grad_E_ss(l,x) gives the ∇ E^l(x), where 1/N ∑_{l=1}^N E^l(x) = E(x), the full potential function
# Q is a symmetric matrix such that - Q ⪯ ∇^2 E^l(x) ⪯ Q for all l,x

    freq_adap = convert(Int64,time_adaps/discr_step)
    if (length(x_ref) == 0 || length(∇E_ref) == 0)
        controlvariates = false;
        error("BPS_Subampling without control variates currently not supported")
    else
        controlvariates = true;
    end

    dim = length(x_ref)
    if length(x_init) == 0
        if controlvariates
            x_init = x_ref
        else
            x_init = zeros(dim)
        end
    end
    if length(v_init) == 0
        v_init = randn(dim)
    end

    t = 0.0;
    n = 0; n_steps = convert(Integer,T/discr_step);
    x = x_init; v = copy(v_init); θ = copy(v_init);
    x_matrix = Array{Float64,2}(undef,dim,(n_steps + 1))
    x_matrix[:,1] = x
    skel_chain = skeleton[]
    push!(skel_chain,skeleton(x,v,t))
    index = 1
    t_limit = time_adaps
    iter = 0
    old_bounces = 0; old_events = 0;
    updateSkeleton = false;
    finished = false;
    rejected_switches = 0;
    accepted_switches = 0;
    n_refresh = 0;
    sqnorm_v = dot(v,v)
    b = hessian_bound_ss * sqnorm_v
    a = dot(v, ∇E_ref) + hessian_bound_ss * sqrt(sqnorm_v) * norm(x-x_ref)
    Δt_switch_proposed = switchingtime(a,b)


    Δt_switch_proposed = switchingtime(a,b)
    if refresh_rate <= 0.0
        Δt_refresh = Inf
    else
        Δt_refresh = -log(rand())/refresh_rate
    end
    if !only_refresh
      if diag_adaptation
          M = ones(dim);
          μ_hat = zeros(dim);
          μ_old = zeros(dim);
          Σ_hat = zeros(dim);

          while (!finished)
              Δt = min(Δt_switch_proposed,Δt_refresh);
              if (t+Δt) < t_limit   # then simulate from current kernel
                  x = x + v * Δt; # O(d)
                  t = t + Δt;
                  a = a + b * Δt; # O(d).
                  if (Δt_switch_proposed < Δt_refresh)
                      # a = a + b * Δt; # O(d).
                      j = rand(1:n_observations)
                      ∇E_est = ∇E_ref + ∇E_ss(j,x) - ∇E_ss(j,x_ref)
                      switch_rate = transpose(v) * ∇E_est
                      proposedSwitchIntensity = a
                      if proposedSwitchIntensity < switch_rate - tolerance
                          println("ERROR: Switching rate exceeds bound.")
                          println(" simulated rate: ", proposedSwitchIntensity)
                          println(" actual switching rate: ", switch_rate)
                          error("Switching rate exceeds bound.")
                      end
                      if rand() * proposedSwitchIntensity <= switch_rate
                          transformed_gradient = M.*∇E_est
                          θ = reflect(transformed_gradient,θ)
                          v = M.*θ
                          updateSkeleton = true
                          accepted_switches += 1
                      else  # reject proposed switch
                          updateSkeleton = false
                          rejected_switches += 1
                      end
                      Δt_refresh = Δt_refresh - Δt_switch_proposed
                  else  # refreshment
                      updateSkeleton = true
                      n_refresh+=1
                      θ = randn(dim)
                      v = M.*θ
                      # update time to refresh
                      Δt_refresh = -log(rand())/refresh_rate;
                  end
              elseif t + Δt < T     # then it is time to update the parameters
                  Δt = (n+1)*time_adaps - t
                  updateSkeleton = true
                  x = x + v * Δt; # O(d)
                  t = (n+1)*time_adaps;
                  x_matrix[:,(iter+2):((n+1)*freq_adap+1)] = discretise(skel_chain[index:end],discr_step, t, dim)
                  μ_hat,Σ_hat = update_γ_diag(x_matrix[:,(iter+2):((n+1)*freq_adap+1)],Σ_hat,μ_hat,μ_old,iter)
                  μ_old = copy(μ_hat)
                  U = rand()
                  if (t/discr_step > dim) && (U <= updating_probs(n+1))
                    M = sqrt.(Σ_hat)
                    v = M.*θ;
                  end
                  # v = M.*θ;
                  n += 1
                  iter = n*freq_adap
                  index = (accepted_switches + n_refresh) + n + 1
                  t_limit += time_adaps
                  if adapt_refresh && (U <= updating_probs(n+1))
                    new_events  = accepted_switches+n_refresh-old_events
                    new_bounces = accepted_switches-old_bounces
                    if (new_bounces)/(new_events) > 1-0.7812
                      refresh_rate = min(10,refresh_rate + 1/(n^2))
                    else
                      refresh_rate = max(0.01, refresh_rate - 1/(n^2))
                    end
                    old_bounces = accepted_switches
                    old_events  = accepted_switches+n_refresh
                  end
                  Δt_refresh = -log(rand())/refresh_rate;
              else   # time to wrap up
                  Δt = T - t # correct to get right to the end of the time horizon
                  finished = true
                  updateSkeleton = true
                  x = x + v * Δt; # O(d)
                  t = T;
                  x_matrix[:,(iter+2):end] = discretise(skel_chain[index:end],discr_step, t, dim)
              end
              if updateSkeleton
                  push!(skel_chain,skeleton(x,v,t))
                  updateSkeleton = false
                  b =  hessian_bound_ss * dot(v,v)
              end
              a = dot(v, ∇E_ref) + sqrt(b) * sqrt(hessian_bound_ss) * norm(x-x_ref)
              Δt_switch_proposed = switchingtime(a,b)
            end

        else  # Learn the full covariance matrix

          M = Symmetric(Matrix{Float64}(I, dim, dim))
          μ_hat = zeros(dim);
          μ_old = zeros(dim);
          Σ_hat = zeros(dim,dim);

          while (!finished)
              Δt = min(Δt_switch_proposed,Δt_refresh);
              if (t+Δt) < t_limit   # then simulate from current kernel
                  x = x + v * Δt; # O(d)
                  t = t + Δt;
                  a = a + b * Δt; # O(d).
                  if (Δt_switch_proposed < Δt_refresh)
                      j = rand(1:n_observations)
                      ∇E_est = ∇E_ref + ∇E_ss(j,x) - ∇E_ss(j,x_ref)
                      switch_rate = transpose(v) * ∇E_est
                      proposedSwitchIntensity = a
                      if proposedSwitchIntensity < switch_rate - tolerance
                          println("ERROR: Switching rate exceeds bound.")
                          println(" simulated rate: ", proposedSwitchIntensity)
                          println(" actual switching rate: ", switch_rate)
                          error("Switching rate exceeds bound.")
                      end
                      if rand() * proposedSwitchIntensity <= switch_rate
                          transformed_gradient = transpose(M)*∇E_est
                          θ = reflect(transformed_gradient,θ)
                          v = M*θ
                          updateSkeleton = true
                          accepted_switches += 1
                      else  # reject proposed switch
                          updateSkeleton = false
                          rejected_switches += 1
                      end
                      Δt_refresh = Δt_refresh - Δt_switch_proposed
                  else  # refreshment
                    n_refresh+=1
                    updateSkeleton = true
                    θ = randn(dim)
                    v = M*θ
                    # update time to refresh
                    Δt_refresh = -log(rand())/refresh_rate;
                  end
              elseif t + Δt < T     # then it is time to update the parameters
                  Δt = (n+1)*time_adaps - t
                  updateSkeleton = true
                  x = x + v * Δt; # O(d)
                  t = (n+1)*time_adaps;
                  x_matrix[:,(iter+2):((n+1)*freq_adap+1)] = discretise(skel_chain[index:end],discr_step, t, dim)
                  μ_hat,Σ_hat = update_γ(x_matrix[:,(iter+2):((n+1)*freq_adap+1)],Σ_hat,μ_hat,μ_old,iter)
                  μ_old = copy(μ_hat)
                  U = rand()
                  if t/discr_step > dim && U <= updating_probs(n+1)
                    # Σ_hat = Symmetric(Σ_hat)
                    if preserve_nr_events
                      # decomposition = eigen(Σ_hat)
                      # determinant = prod(decomposition.values)
                      # M = decomposition.vectors * ((1/determinant * diagm(decomposition.values)).^(1/2)) * transpose(decomposition.vectors)
                      M = sqrt(dim/tr(Σ_hat)) * Symmetric(sqrt(Σ_hat))
                      v = M*θ;
                    else
                      # M = convert(Array{Float64,2},sqrt(Σ_hat))
                      M = Symmetric(sqrt(Σ_hat))
                      v = M*θ;
                      # M = convert(Symmetric{Float64,Array{Float64,2},sqrt(Σ_hat))
                    end
                  end
                  # v = M*θ;
                  n += 1
                  iter = n*freq_adap
                  index = (accepted_switches + n_refresh) + n + 1
                  t_limit += time_adaps
                  if adapt_refresh && (U <= updating_probs(n+1))
                    new_events  = accepted_switches+n_refresh-old_events
                    new_bounces = accepted_switches-old_bounces
                    if (new_bounces)/(new_events) > 1-0.7812
                      # refresh_rate = min(10,refresh_rate + 1/log(t))
                      refresh_rate = min(10,refresh_rate + 1/(n^2))
                    else
                      # refresh_rate = max(0.01, refresh_rate - 1/log(t))
                      refresh_rate = max(0.01, refresh_rate - 1/(n^2))
                    end
                    old_bounces = accepted_switches
                    old_events  = accepted_switches+n_refresh
                  end
                  Δt_refresh = -log(rand())/refresh_rate;
              else   # time to wrap up
                  Δt = T - t # correct to get right to the end of the time horizon
                  finished = true
                  updateSkeleton = true
                  x = x + v * Δt; # O(d)
                  t = T;
                  x_matrix[:,(iter+2):end] = discretise(skel_chain[index:end],discr_step, t, dim)
              end
              if updateSkeleton
                  push!(skel_chain,skeleton(x,v,t))
                  updateSkeleton = false
                  b =  hessian_bound_ss * dot(v,v) # = ||C||_2 * || Mθ||_2^2
              end
              a = dot(v, ∇E_ref) + sqrt(b) * sqrt(hessian_bound_ss) * norm(x-x_ref)
              Δt_switch_proposed = switchingtime(a,b)
            end #close while
          end #close if diag adap or full adap
        else # then we want to adapt only the refreshment rate
          adapt_refresh = true
          while (!finished)
              Δt = min(Δt_switch_proposed,Δt_refresh);
              if (t+Δt) < t_limit   # then simulate from current kernel
                  x = x + v * Δt; # O(d)
                  t = t + Δt;
                  a = a + b * Δt; # O(d).
                  if (Δt_switch_proposed < Δt_refresh)
                      j = rand(1:n_observations)
                      ∇E_est = ∇E_ref + ∇E_ss(j,x) - ∇E_ss(j,x_ref)
                      switch_rate = transpose(v) * ∇E_est
                      proposedSwitchIntensity = a
                      if proposedSwitchIntensity < switch_rate - tolerance
                          println("ERROR: Switching rate exceeds bound.")
                          println(" simulated rate: ", proposedSwitchIntensity)
                          println(" actual switching rate: ", switch_rate)
                          error("Switching rate exceeds bound.")
                      end
                      if rand() * proposedSwitchIntensity <= switch_rate
                          gradient = ∇E_est
                          v = reflect(gradient,v)
                          updateSkeleton = true
                          accepted_switches += 1
                      else  # reject proposed switch
                          updateSkeleton = false
                          rejected_switches += 1
                      end
                      Δt_refresh = Δt_refresh - Δt_switch_proposed
                  else  # refreshment
                    n_refresh+=1
                    updateSkeleton = true
                    v = randn(dim)
                    # update time to refresh
                    Δt_refresh = -log(rand())/refresh_rate;
                  end
              elseif t + Δt < T     # then it is time to update the parameters
                  Δt = (n+1)*time_adaps - t
                  updateSkeleton = true
                  x = x + v * Δt; # O(d)
                  t = (n+1)*time_adaps;
                  n += 1
                  t_limit += time_adaps
                  U = rand()
                  if adapt_refresh && (U <= updating_probs(n+1))
                    new_events  = accepted_switches+n_refresh-old_events
                    new_bounces = accepted_switches-old_bounces
                    if (new_bounces)/(new_events) > 1-0.7812
                      refresh_rate = min(10,refresh_rate + 1/(n^2))
                    else
                      refresh_rate = max(0.01, refresh_rate - 1/(n^2))
                    end
                    old_bounces = accepted_switches
                    old_events  = accepted_switches+n_refresh
                  end
                  Δt_refresh = -log(rand())/refresh_rate;
              else   # time to wrap up
                  Δt = T - t # correct to get right to the end of the time horizon
                  finished = true
                  updateSkeleton = true
                  x = x + v * Δt; # O(d)
                  t = T;
                  x_matrix[:,(iter+2):end] = discretise(skel_chain[index:end],discr_step, t, dim)
              end
              if updateSkeleton
                  push!(skel_chain,skeleton(x,v,t))
                  updateSkeleton = false
                  b =  hessian_bound_ss * dot(v,v) # = ||C||_2 * || Mθ||_2^2
              end
              a = dot(v, ∇E_ref) + sqrt(b) * sqrt(hessian_bound_ss) * norm(x-x_ref)
              Δt_switch_proposed = switchingtime(a,b)
          end
        end
    println("ratio of accepted switches: ", accepted_switches/(accepted_switches+rejected_switches))
    println("number of proposed switches: ", accepted_switches + rejected_switches)

    if return_positions
      return (x_matrix,skel_chain)
    else
      return skel_chain,accepted_switches,rejected_switches
    end

end
