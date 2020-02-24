# updateRule.jl

mutable struct UpdateRule
    update # function that updates the current parameters
end

### Perfect integration
function perfect()

    function update(t, transitions, alpha0)
        # update to Dirichlet prior is simply adding the occurences to alpha
        allTransitions = dropdims(sum(transitions, dims = 3), dims = 3)
        return alpha0 + allTransitions
    end

    return UpdateRule(update)
end

# Leaky integration
function leaky(w)

    function update(t, transitions, alpha0)
        # first we weigh each transition by how far in the past it is
        decay = exp.(-1.0 / w * (t:-1:1))
        weightedTransitions = decay .* transitions

        # update to Dirichlet prior is simply
        # adding the weighted occurences to alpha
        allTransitions = dropdims(sum(weightedTransitions, dims = 3), dims = 3)
        return alpha0 + allTransitions
    end

    return UpdateRule(update)
end

# Variational SMiLe
function varSMiLe(m)
    # for now, this is the same as a perfect integrator
    return perfect()
end
