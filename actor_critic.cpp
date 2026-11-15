#include "actor_critic.h"


/*********************************************************/
/* -------------------- ACTOR CRITIC ------------------- */
/*********************************************************/

AInfo* SInfo::add_action(int _a, float _theta, Action _action) {
    auto [it, inserted] = actions.emplace(_a, AInfo(_a));
    if (inserted) { it->second.theta = _theta; it->second.ref = _action; }
    return &it->second;
}

ActorCritic::ActorCritic(System& _system, bool _stoch):system(_system) {
    record_state(*system.state_zero);
}

/* Try to emplace if not already present*/
SInfo& ActorCritic::record_state(SystemState& state) {
    string key = system.hash_state(state);
    return record_state(key, state);
}

SInfo& ActorCritic::record_state(string &key, SystemState& state) {
    
    auto [it, inserted] = states.emplace(key, SInfo(key));
    SInfo* sinfo = &it->second;
    sinfo->visited++;    
    if (inserted) {
        
        Actions actions;        
        system.get_available_actions(state, actions);               
        for (auto& action : actions.info) {
            int a = system.get_action_index(action);
            AInfo *ptr = sinfo->add_action(a, 0.0f, action);                        
        }   
        update_policy(state);
    }
   
    return it->second;
}

/* Take an action, transition system, compute reward */
void ActorCritic::step(SystemState& state, Action& action, SystemState& next, float& reward) {
    next.copy(state);
    system.apply_action(next, action);
    
    auto next_point = system.go_to_next_decision_point(next);
    record_state(next); // if unseen
  
    if (!next_point) next.is_terminal = true; 
    reward = system.compute_reward(state, next, action);   
}

/* Extract the Info record  */
SInfo* ActorCritic::get_info(SystemState& state) {
    string key = system.hash_state(state);
    return get_info(key);
}

SInfo* ActorCritic::get_info(const std::string& key) {
    auto it = states.find(key);
    if (it != states.end()) return &(*it).second;
    return nullptr;
}

void ActorCritic::update_theta(SystemState& state, Action& action, float delta) {
    int chosen = system.get_action_index(action);
    SInfo* info = get_info(state);

    for (auto& action : info->actions) {
        int a = action.first;   
        AInfo* ptr = &action.second;
        float grad;
        if (a == chosen)
            grad = (1.0f - ptr->prob);
        else
            grad = -ptr->prob;
            
        ptr->theta += alpha_actor * delta * grad;

        /// Optional. Benefit is unproven.
        //ptr->theta += entropy * (-log(ptr->prob + 1e-8f));
    }
}

/* Refine actors policy */
void ActorCritic::actor_update(SystemState& state, Action& action, float delta) {  
    update_theta(state, action, delta); // Compute theta
    update_policy(state); // Revise probabilities ?
}

void ActorCritic::add_to_pool(string key, float score) { pool.insert(key); }

void ActorCritic::add_to_pool(SystemState& state) {
    SInfo* info = get_info(state);
    add_to_pool(system.hash_state(state), info->interest);
}

/* Stabilized softmax */
void ActorCritic::update_policy(SystemState& state) {
    SInfo* info = get_info(state);
   
    // Compute max_theta for numerical stability
    float max_theta = -infinity;
    for (auto& action : info->actions) {
        AInfo* ptr = &action.second;
        max_theta = max(max_theta, ptr->theta);
    }

    // Denominator calculation
    float denom = 0.0f;
    for (auto& action : info->actions) {
        AInfo* ptr = &action.second;
        denom += exp(ptr->theta - max_theta);
    }

    // Probability calculation
    for (auto& action : info->actions) {
        AInfo* ptr = &action.second;
        ptr->prob = exp(ptr->theta - max_theta) / denom;
    }
}

/* */
float ActorCritic::compute_delta(SInfo& state, SInfo& next, float reward, bool next_is_terminal) {
    if (next_is_terminal) return reward - state.value;
    else return reward + gamma * next.value - state.value;
}

float ActorCritic::compute_delta(SystemState& state, SystemState& next, float reward) {
    SInfo* si = get_info(state);
    SInfo* ni = get_info(next);
    return compute_delta(*si, *ni, reward, next.is_terminal);    
}

/* Choose an action using the policy */
Action ActorCritic::sample_action(SystemState& state) {
    SInfo* si = get_info(state);

    std::vector<AInfo*> actions;
    std::vector<double> weights;

    actions.reserve(si->actions.size());
    weights.reserve(si->actions.size());

    bool all_zero = true;
    for (auto& element : si->actions) {
        AInfo* ptr = &element.second;
        actions.push_back(ptr);
        weights.push_back(ptr->prob);
        all_zero &= (ptr->prob < 1e-12);
    }

    /// Create distributions
    std::uniform_real_distribution<> urand(0, 1);
    int choice;
    if (all_zero or urand(system.gen) < epsilon) {
        std::uniform_int_distribution<> unif(0, static_cast<int>(actions.size()) - 1);
        choice = unif(system.gen); // Sample      
    }
    else {
        std::discrete_distribution<> dist(weights.begin(), weights.end());
        choice = dist(system.gen); // Sample      
    }
    return actions[choice]->ref;
}


/* Refine (.a.k.a. update) the state value - three variants */
void ActorCritic::critic_update(SInfo& info, float delta) {
    info.value += alpha_critic * delta;
}

void ActorCritic::critic_update(string& key, float delta) {
    auto it = states.find(key);
    if (it == states.end()) throw std::runtime_error("No state with specified key");
    critic_update(it->second, delta);
}

void ActorCritic::critic_update(SystemState& state, float delta) {
    string key = system.hash_state(state);
    critic_update(key, delta);
}

/* Store the transition info */
void ActorCritic::store_transition(SystemState& state, Action& action, SystemState& next, float reward) {       
    buffer.emplace_back(BufferInfo(state, next, action, reward, next.is_terminal));
    if (buffer.size() > replay_buffer) buffer.pop_front();
}

/// Scheduling problems have many uniformly bad states, and a few promising states
/// Softmax replay naturally amplifies the best regions while suppressing low - quality regions

void ActorCritic::random_replay() {
    if (buffer.size() < replay_buffer) return;

    std::uniform_int_distribution<> unif(0, static_cast<int>(buffer.size()) - 1);
    for (int replay = 0; replay < nreplay; replay++) {
        int index = unif(system.gen); // Choose a transition
        BufferInfo& info = buffer[index]; // Transition record   
        delta = compute_delta(info.state, info.next, info.reward);
        critic_update(info.state, delta);
        actor_update(info.state, info.action, delta);
    }
}

/* */
void ActorCritic::random_replay_rollout() {
    std::uniform_int_distribution<> unif(0, static_cast<int>(buffer.size()) - 1);
    for (int replay = 0; replay < nreplay; replay++) {
        int index = unif(system.gen); // Choose a transition
        BufferInfo& info = buffer[index]; // Transition record   
        system.restore_state(info.state);
        step(info.state, info.action, info.next, info.reward); 
        sequential_rollouts(&info.next);
    }
}

/* Choose replays based upon interest level */
/// EXTRACTING WEIGHTS IS TOO SLOW!
void ActorCritic::prioritized_replay() {    
    if (buffer.size() < replay_buffer) return;
   
    ///  Extract weights
    float delta;
    vector<float> weights;
    weights.reserve(buffer.size());
    for (auto& transition : buffer) {
        SInfo* info = get_info(transition.state); 
        float w = info ? info->interest : -1e6f;
        if (transition.next.is_terminal) w *= 3;
        weights.push_back(w);
    }

    /// Scale weights
    float max_weight = *max_element(weights.begin(), weights.end());
    float T = 1; // temperature
    for (float& weight : weights) weight = exp((weight - max_weight) / T);

    /// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /// Do replays
    std::discrete_distribution<> dist(weights.begin(), weights.end());
    for (int replay = 0; replay < nreplay; replay++) {
        int index = dist(system.gen); // Sample
        BufferInfo& info = buffer[index]; // Transition record   
        delta = compute_delta(info.state, info.next, info.reward);
        critic_update(info.state, delta);
        actor_update(info.state, info.action, delta);                    
    }
}

/* */
SystemState ActorCritic::choose_state(float prob) {
    int sz = static_cast<int>(pool.size());

    /// No options other than state_zero
    /// or choose state zero by probability
    std::bernoulli_distribution choose(prob);
    if (sz == 0 or choose(system.gen)) return *system.state_zero;
  
    /// Choose according to interest level
    std::vector<double> weights;
    weights.reserve(sz);
    std::vector<std::string> keys;
    keys.reserve(sz);
    for (const auto& key : pool) {
        keys.push_back(key);
        auto it = states.find(key);
        double w = (it != states.end()) ? it->second.interest : 0.0;
        weights.push_back(w);
    }

    bool all_zero = std::all_of(weights.begin(), weights.end(), [](double w) { return w == 0.0; });
    int index;
    if (all_zero) {
        std::uniform_int_distribution<> uniform_dist(0, sz - 1);
        index = uniform_dist(system.gen);
    }
    else {
        std::discrete_distribution<> dist(weights.begin(), weights.end());
        index = dist(system.gen);
    }
    SystemState state = system.unhash_state(keys[index]);   
    return state;
}

/* Perform one rollout */
void ActorCritic::off_policy_rollout(SystemState & state, int H) {
    SimResult result;
    system.simulate_from(state, result, H); // Simulate the future  
    Trajectory trajectory = system.make_trajectory(state, result.actions_log, result.sim_end);
    process_trajectory(trajectory);
}

/* Perform multiple rollouts in parallel */
/// Current issue: While simulations can be performed in parallel, updates cannot.
/// Hence no real gains can be achieved.
void ActorCritic::parallel_off_policy_rollouts(SystemState &state, int H, int nthrds) {
    vector<Trajectory> trajectories = system.batch_simulate(nroll, nthrds, H, state);
    Trajectory* best_trajectory = nullptr;
    float max = -infinity;
    for (int i = 0; i < nroll; i++) {
        trajectories[i].value = get_discounted_reward(trajectories[i].transitions);
        process_trajectory(trajectories[i]);
        
        // Alternative: Identify best trajectory 
        //for (auto& trajectory : trajectories) { 
        // if (trajectory.value > max) { max = trajectory.value; best_trajectory = &trajectory; }
    }
    //Alternative: process_trajectory(*best_trajectory);// Process best trajectory
}

/* Future extension */
void ActorCritic::parallel_on_policy_rollouts(SystemState& state, int H, int nthrds) {} // TODO

/* Perform multiple rollouts sequentially */
// This approach is memory intensive & registers every state observed in each rollout 
void ActorCritic::sequential_rollouts(SystemState* starting_state) {
    int steps;
    steps = (!rollout_horizon) ? INT_MAX : *rollout_horizon;
    for (int roll = 0; roll < nroll; roll++) {
        SystemState state = (starting_state) ? *starting_state : choose_state();
        if (!state.is_terminal) {
            if (on_policy) on_policy_rollout(state, steps);
            else off_policy_rollout(state, steps);
        }
    }
}

/* */
void ActorCritic::parallel_rollouts(SystemState* starting_state) {
    int steps;
    steps = (!rollout_horizon) ? INT_MAX : *rollout_horizon;

    SystemState state = (starting_state) ? *starting_state : choose_state();
    if (!state.is_terminal) {
        if (on_policy) 
            parallel_on_policy_rollouts(state, steps, *nthrds);
        else 
            parallel_off_policy_rollouts(state, steps, *nthrds);
    }
}

// Simulate "on-policy" to understand the future better - No updates version
float ActorCritic::on_policy_lookahead(SystemState& state, int nsim, int steps) {
    vector<float> G(nsim, 0.0f);
    //#pragma omp parallel for
    for (int n = 0; n < nsim; n++) {
        SystemState current = state;
        SystemState next;
        float reward, gamma_t = 1.0f, gamma = 0.99f;
        for (int h = 0; h < steps; h++) {
            Action action = sample_action(current);
            step(current, action, next, reward);
            G[n] = reward + gamma_t * G[n];
            gamma_t *= gamma;
            current.copy(next);
            if (current.is_terminal) break;
        }
    }
    float avg_G = 0.0f;
    for (int n = 0; n < nsim; n++) avg_G += G[n];
    avg_G = avg_G / nsim;
    float delta = avg_G - get_info(state)->value;
    return avg_G;
}

/* Sample actions from the policy and perforn immediate (one-step) updates  at each step */
void ActorCritic::on_policy_rollout_one(SystemState& state, int steps) {
    SystemState current = state;
    SystemState next;
    float reward;
    for (int h = 0; h < steps; h++) {
        Action action = sample_action(current); // Choose an action to take
        step(current, action, next, reward);// Transition the system. Record the new state
        float delta = compute_delta(current, next, reward);
        update_interest(current, next, reward, delta);
        critic_update(current, delta);
        actor_update(current, action, delta);
        current.copy(next);
        if (current.is_terminal) break;
    }
}

/* Sample actions from the policy and perforn multiple updates */
void ActorCritic::on_policy_rollout(SystemState& state, int H) {  
     
    SystemState next, current = state;
    float reward;
    
    std::vector<SystemState> states;
    std::vector<Action> actions;
    std::vector<float> rewards;

    states.reserve(H);
    actions.reserve(H);
    rewards.reserve(H);

    /// Perform rollouts - sample and record dynamics
    for(int h = 0; h < H; h++) {
        Action action = sample_action(current);        
        step(current, action, next, reward); // Record the new state
        states.emplace_back(current); // Make a copy
        actions.emplace_back(action); // Make a copy
        rewards.emplace_back(reward);  // Make a copy
        current.copy(next);
        if (current.is_terminal) break;
    }

    /// Perform discounted reward calculations - before updates
    // Note: get_info(states[i])->value changes after critic updates
    // Hence: Work out all deltas first!
    // Pros: more stable, lower variance. Cons: more memory
    float G = (next.is_terminal) ? 0.0f :  get_info(next)->value;
    std::vector<float> delta(states.size());
    for (int h = (int) states.size(); h-- > 0; ) { // backwards iteration
        G = rewards[h] + gamma * G;
        delta[h] = G - get_info(states[h])->value;
    }

    /// Perform Batch Updates
    for (int h = 0; h < (int) states.size(); ++h) {
        critic_update(states[h], delta[h]);
        actor_update(states[h], actions[h], delta[h]);
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Optional "all-in-one" update phase. 
    // Pros: simple; 
    // Cons: Inconsistent critic. Later states use a partially updated critic
    // Cons: moving targets, noisier learning, rollout updates depend on processing order
    /* float delta, G = 0.0f;
    if (!next.is_terminal) G = get_info(next)->value;
    for (int h = (int) states.size() - 1; h-- > 0) {
        G = rewards[h] + gamma * G; // Discount the future reward
        delta = G - get_info(states[h])->value; // Compute the advantage
        critic_update(states[h], delta);
        actor_update(states[h], actions[h], delta);      
    }*/   
}

/* */
float ActorCritic::get_discounted_reward(vector<Transition>& trans) {
    float gamma_t = 0.9f; 
    string &key = (trans.back()).hash_next_state;
    SystemState final = system.unhash_state(key);
    float G = (final.is_terminal) ? 0.0f : get_info(key)->value;
    for (int h = (int) trans.size() - 1; h >= 0; --h)
        G = trans[h].reward + gamma_t * G;    
    return G;
}


/* */
void ActorCritic::process_trajectory(Trajectory &trajectory) {
    
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /// Initialise vectors
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    int sz = (int) trajectory.transitions.size();
    vector<Transition>& trans = trajectory.transitions;
    vector<SystemState> states;
    vector<Action> actions;
    states.reserve(sz + 1);
    actions.reserve(sz);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /// Unhash and register observed states
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    SystemState _state = system.unhash_state(trans[0].hash_state);
    states.push_back(_state);
    record_state(states.back()); // register state here
    for (const auto& t : trans) {
        SystemState _state = system.unhash_state(t.hash_next_state);
        states.push_back(_state);
        record_state(states.back()); // register state here
        Action action = system.unhash_action(t.hash_action);
        actions.push_back(action);
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /// Compute advantages
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    float gamma_t = 0.9f;
    vector<float> advantage(sz);
    string& key = (trans.back()).hash_next_state;
    SystemState final = system.unhash_state(key);
    float G = (final.is_terminal) ? 0.0f : get_info(key)->value;
    for (int h = (int) trans.size(); h-- > 0; ) { // backwards iteration
        G = trans[h].reward + gamma_t * G;
        advantage[h] = G - get_info(trans[h].hash_state)->value; // future value - current value of state
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /// Batch Updates
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for (int h = 0; h < (int) states.size() - 1; ++h) {
        critic_update(states[h], advantage[h]);
        actor_update(states[h], actions[h], advantage[h]);
    }
}

/* */
void ActorCritic::off_policy_rollout_one(SystemState& state, int H) { 
    SimResult result;
    system.simulate_from(state, result, H); // Simulate the future - H steps
    Trajectory trajectory = system.make_trajectory(state, result.actions_log, result.sim_end); // Assemble trajectory information
    vector<Transition>& records = trajectory.transitions;
    float R = get_discounted_reward(records);
    string key = records.back().hash_next_state;
    SystemState final = system.unhash_state(key);
    auto* info = get_info(records[0].hash_state);
    auto* final_info = get_info(key);
    delta = R - info->value;
}

void ActorCritic::report(ostream& out) {
    out << "\n#Episodes: " << episode;
    out << "\nAlpha:" << alpha_critic;
    out << "\nGamma:" << gamma;
    out << "\nBuffer:" << replay_buffer;
    out << "\nReplays:" << nreplay;
    out << "\nRollouts:" << nroll;
    out << "\n#States = " << states.size();
    out << "\n#TerminalStates = " << terminal_states.size();
    out << "\nBest score: " << best_score;
    out << "\nWall time: " << wall_time;
    out << "\nActions: ";
    for (auto& action : best_actions)
        out << "\n" << system.hash_action(action) << ",";   
}

/* */
void ActorCritic::update_best(vector<Action>& actions, float score) {
    if (score > best_score) {       
        best_actions = actions; // Overwrite
        best_score = score;        
    }
}

/* delta = reward + future value - current value */
void ActorCritic::update_interest(SystemState& state, SystemState& next, float reward, float delta, int method) {
    SInfo* info = get_info(state);
    if (!info) return; // Safety check
   
    // Basic approach - reward + future value 
    if (method == 1) {
        SInfo* next_info = get_info(next);
        info->interest = reward + next_info->value;
    }

    // Scale negative values
    if (method == 2) { // If state is poor, but interesting, make positive and reduce magnitude of delta
        float adjustment = (delta > 0.0f) ? delta : 0.1f * fabsf(delta); // 0.1f is the scaling factor
        info->interest += adjustment;
    }

    // Weight earlier level more   
    if (method == 3) {
        float beta = 0.05f;
        info->interest = (1.0f - beta) * info->interest + beta * max(0.0f, delta);  
    }

    // // Guard against negativity - interest is positive or zero
    if (method == 4) { info->interest = max(0.0f, delta);  }
}

bool ActorCritic::is_interesting(float delta) { return (abs(delta) > 1.5 * mean_abs_delta); }

/* Generate a single trajectory  */
SystemState ActorCritic::iterate(vector<Action> &actions) {
    SystemState next, state = choose_state(0.2f); // 20% chance of state_zero
    system.restore_state(state);
    float reward;   

    while (!state.is_terminal) {

        string key = system.hash_state(state);
        
        /// Select an action
        Action action = sample_action(state); 
        string action_code = system.hash_action(action);
        action.when = state.time;
        actions.push_back(action);

        /// Identify next state and record, observe reward               
        step(state, action, next, reward);
        string key_next = system.hash_state(next);

        /// Calculation of delta and revision of mean
        delta = compute_delta(state, next, reward);
        trans++; // # of transitions performed
        mean_abs_delta += abs(delta - mean_abs_delta) / trans ;

        /* Note: delta > 0 => outcome better than expected
           States that are much better and much worse are 
           informative for replay prioritization 
        */

        /// Update actor policy and critic value functions - Core updates
        critic_update(state, delta);
        actor_update(state, action, delta);

        /// Replays
        if (nreplay > 0) {            
            if(is_interesting(delta)) 
                store_transition(state, action, next, reward); // Store interesting transitions
            random_replay();
        }

        /// Rollouts 
        if (nroll > 0 and !next.is_terminal) {
            if(nthrds) parallel_rollouts(&next);
            else sequential_rollouts(&next);            
        }
  
        /// Revise interest level and store state
        // update_interest(state, next, reward, delta);
        //if (!state.is_terminal and abs(delta) > threshold) add_to_pool(state);
      
        state.copy(next);
    }
   
    return state;
}

void ActorCritic::record_terminal_state(SystemState &state) {
    string key = system.hash_state(state);
    terminal_states.insert(key);    
}

/* Basic Implementation with Prioritized Replay and Rollouts */
void ActorCritic::run(int episodes) {
  
    epsilon = 0.1f; float decay = 1.0f; // Static epsilon

    /// For off-policy simulations
    bool par = (!on_policy and nthrds); // Go parallel
    if (nroll > 0 and par) {
        system.setup_components(*nthrds, system.gen());
        cout << endl << "Hi";
    }

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // VARIABLE EPSILON. epsilon(t) = epsilon_initial * exp(- decay * t) 
    // float epsilon = 0.5f, final_epsilon = 0.05f;
    // float decay = exp((log(final_epsilon) - log(epsilon)) / episodes);
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    wall_start = omp_get_wtime();
    for (episode = 0; episode < episodes; episode++) {
        vector<Action> actions;
        SystemState final_state = iterate(actions);
        record_terminal_state(final_state);  
        float score = system.evaluate_actions(actions, false);
        update_best(actions, score);
        cout << "\n" << episode << " -> " << best_score; // Report
        epsilon *= decay; // Decay the epsilon value       
    }
   
    wall_end = omp_get_wtime();
    wall_time = wall_end - wall_start;
}




