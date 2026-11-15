#include "deslearn.h"

using namespace std;

extern std::mt19937 rng;

extern int NTHRDS;

/* Lexicographic dominance for maximization */
bool dominates(Point* pt_1, Point* pt_2) {
    bool strictly_better = false;
    for (int i = 0; i < pt_1->size(); i++) {
        if ((*pt_1)[i] < (*pt_2)[i]) return false;
        if ((*pt_1)[i] > (*pt_2)[i]) strictly_better = true;
    }
    return strictly_better;
}

/* Assumption: final state is a terminal state */
float get_discounted_reward(vector<Transition>& trans) {
    float G = 0.0f, gamma = 1.0f;// 0.9f;
    for (int h = (int)trans.size() - 1; h >= 0; --h)
        G = trans[h].reward + gamma * G;
    return G;
}

/* Simulate till completion or for a given number of steps */
Trajectory get_simulated_trajectory(System& system, SystemState& state, optional<int> steps) {
    SimResult result;
    system.simulate_from(state, result, steps);
    Trajectory trajectory = system.make_trajectory(state, result.actions_log, result.sim_end);
    trajectory.value = get_discounted_reward(trajectory.transitions);
    return trajectory;
}

vector<Trajectory> batch_simulate(int nsim, int nthrds,  System& system, SystemState& state, optional<int> steps) {
    vector<Trajectory> trajectories = system.batch_simulate(nsim, nthrds, *steps, state);    
    for (int i = 0; i < nsim; i++) {
        trajectories[i].value = get_discounted_reward(trajectories[i].transitions);        
    }
    return trajectories;
}

/*********************************************************/
/* --------- REINFORCEMENT LEARNING AGENT CLASS -------- */
/*********************************************************/

/* Constructor */
ReinfLearnAgent::ReinfLearnAgent(ReinfLearn& _rl) :rl(_rl) {
    current_reward = 0;
    current_state = nullptr;
    previous_state = nullptr;
    update_fn = nullptr;

    // Define default hyperparameters
    lr = lr_I = 0.5f; lr_decay = 0.01f;
    er = er_I = 0.5f; er_decay = 0.01f;
    df = df_I = 0.5f; df_decay = -0.01f;

    decay = DecayType::linear;
    mode = ExplorationMode::e_greedy;

    update_fn = (decay == DecayType::linear) ? &ReinfLearnAgent::revise_linear : &ReinfLearnAgent::revise_expon;
}

/* Destructor */
ReinfLearnAgent::~ReinfLearnAgent() {
    if (previous_state != nullptr) delete previous_state;
    if (current_state != nullptr) delete current_state;
}

/* */
void ReinfLearnAgent::set_parameters(float _lr_I, float _lr_decay, float _er_I, float _er_decay, float _df_I, float _df_decay) {
    lr = lr_I = _lr_I;
    lr_decay = _lr_decay;
    er = er_I = _er_I;
    er_decay = _er_decay;
    df = df_I = _df_I;
    df_decay = _df_decay;
}

void ReinfLearnAgent::reset_parameters() {
    lr = lr_I;
    er = er_I;
    df = df_I;
}

/* Revise prameters */
void ReinfLearnAgent::revise_linear() {
    er -= er_decay;
    lr -= lr_decay;
    df -= df_decay;
}

void ReinfLearnAgent::revise_expon() {
    lr = lr_I / (1.0f + lr_decay * rl.episode);
    er = er_I / (1.0f + er_decay * rl.episode);
    df = df_I / (1.0f + df_decay * rl.episode);
}

void ReinfLearnAgent::update_parameters() { (this->*update_fn)(); }

/* Using the Q values, choose an action to take at the current state */
Action ReinfLearn::get_best_action(SystemState& state) {
    string hash_id = system.hash_state(state);
    SystemStateInfo* details = get_state_info(hash_id);
    vector<float>& values = details->q;
    float best = -infinity;
    int a = 0, best_at = 0;
    for (auto value : values) {
        if (value > best) { best_at = a; best = value; }
        a++;
    }
    return system.get_action(best_at);
}

/* Select a feasible action to perform at the current state */
Action ReinfLearnAgent::select_action(Actions &actions) {
    switch (mode)
    {
    case ExplorationMode::e_greedy: return epsilon_greedy(actions);
    case ExplorationMode::pareto: return pareto_selection(actions);
    default:
        throw std::runtime_error("Unknown exploration mode");
    }
}

/* Pareto Based Selection */
Action ReinfLearnAgent::pareto_selection(Actions& actions) {
    bool dom;
    Point pt_1, pt_2;
    vector<Action> pareto;
    for (auto action_1 : actions.info) 
    {
        string hash_1 = rl.system.hash_action(action_1); 
        dom = false;
        pt_1 = rl.get_status(current_state, action_1);
        for (auto action_2 : actions.info) 
        {
            string hash_2 = rl.system.hash_action(action_2); 

            if (hash_1 == hash_2) continue; // i.e. same action, so skip
            
            pt_2 = rl.get_status(current_state, action_2); 
             if (dominates(&pt_2, &pt_1)) {                
                dom = true;
                break; // action_1 is dominated => no need to compare to other actions
             }          
        }      
        if (!dom) { pareto.push_back(action_1); }
    }

    if (pareto.empty()) {
        std::uniform_int_distribution<> unif(0, static_cast<int>(actions.info.size()) - 1);
        return actions.info[unif(rl.gen)];
    }
   
    std::uniform_int_distribution<> unif(0, static_cast<int>(pareto.size()) - 1);
    return pareto[unif(rl.gen)];
}

/* E-greedy approach */
Action ReinfLearnAgent::epsilon_greedy(Actions& actions)
{
    Action action;
    std::uniform_real_distribution<> distr(0, 1);
    float rv = (float) distr(rl.gen);

    // ___________________________________________________
    // CHOOSE AN ACTION RANDOMLY
    // ___________________________________________________
    if (rv < er) {
        std::uniform_int_distribution<> unif(0, static_cast<int>(actions.info.size()) - 1);
        int choice = unif(rl.gen);
        action = actions[choice];
    }

    // ___________________________________________________
    // CHOOSE AN ACTION USING PAST EXPERIENCE
    // ___________________________________________________
    else {
        float score, best_score = -infinity;
        int a = 0, best_at = 0;
        for (auto _action : actions.info) {

            score = rl.get_Q(*current_state, _action);
            if (score > best_score) { best_score = score; best_at = a; }
            a++;
        }        
        action = actions[best_at];
    }    
    return action;
}

/* */
void ReinfLearnAgent::reset(SystemState &starting_state) {
    actions.clear();
    trajectory.clear();

    if (previous_state == nullptr) previous_state = new SystemState;
    if (current_state == nullptr) current_state = new SystemState;

    current_state->copy(starting_state); // Copy the starting state
    
    previous_state->records.clear();
    previous_state->is_terminal = false;
    previous_state->time = 0;        
}

/* Take an action form the current state */
optional<SimTime> ReinfLearn::update_system(SystemState &previous, SystemState &current, Action &action) {
    previous.copy(current); // Define the previous state as the current state
    auto next_point = system.transition(current, action);
    return next_point;
}

/* Take an incremental step, i.e., choose an action, identify next time point, and transition system to a new state */
bool ReinfLearnAgent::step() {
    if (current_state == nullptr) throw std::runtime_error("Error. Current state must exist");
    // --------------------------------------------------------
    Actions current_actions;
    rl.system.get_available_actions(*current_state, current_actions); // Find possible actions
    Action action = select_action(current_actions);  // Choose an action
    return take_action(*current_action);    
}

/* Take an action from the current state and update system */
/* Note: previous_state is updated to current_state before the action is applied */
bool ReinfLearnAgent::take_action(Action& action) {   
    current_action = action;
    action.when = current_state->time;
    optional<SimTime> time = rl.update_system(*previous_state, *current_state, *current_action); 
    return (time != nullopt);
}

/* */
float ReinfLearnAgent::compute_accumulated_reward() {
    float G = 0;
    for (auto& transitions: trajectory.transitions | std::views::reverse)
        G += transitions.reward;
    return G;
}



/*********************************************************/
/* ---------------- STATE DETAILS CLASS ---------------- */
/*********************************************************/

SystemStateInfo::SystemStateInfo(string _key, int nb_actions, bool stoch)
    :key(_key), depth(0), created(-1), last_access(-1), visited(0) {
    initialise_arrays(nb_actions, stoch);
}

/* */
void SystemStateInfo::initialise_arrays(int nb_actions, bool stoch) {
    if (!stoch) {
        q.resize(nb_actions, 0); // Originally set as -infinity
    }
    else {
        qmean.resize(nb_actions, 0.0);
        qvar.resize(nb_actions, 0.0);
    }

    prob.resize(nb_actions, 0);
    theta.resize(nb_actions,0);
    std::uniform_real_distribution<> distr(0, 0.1);
    for (int a = 0; a < nb_actions; a++) {
        theta[a] = (float) distr(rng); // random small value
    }

    count.resize(nb_actions, 0);
}

SystemStateInfo::~SystemStateInfo() {
    q.clear();
    qmean.clear();
    qvar.clear();
    count.clear();
}

/*********************************************************/
/* ----------- REINFORCEMENT LEARNING CLASS ------------ */
/*********************************************************/

/* Define default behaviour
 
    Discount: Affects how much future rewards are taken into account
    0 => Look for short term gains. Completely ignore future rewards, immediate rewards are more important
    1 => Look for long term gains.

    Learning Rate:
    0 => Q-value is not updated / no learning 
    1 => Ignore previous Q-value / Adopt the new estimate fully

    Explore Rate:
    1 => Generate random action
    0 => Choose action using experience
*/

/* Set default hyperparameters */
ReinfLearn::ReinfLearn(System& _system, std::mt19937& _gen, bool _stoch) 
    :gen(_gen), system(_system), stoch(_stoch), agent(*this) {
    state_counter = 0;
    mode = ExplorationMode::e_greedy;
    set_default_parameters(INT_MAX);
    register_state(*system.state_zero);
    updateQfn = (stoch) ? &ReinfLearn::updateQ_stoch : &ReinfLearn::updateQ_det;
}

void ReinfLearn::set_default_parameters(int max_episodes) {
    set_lr_range(1.0f, 0.5f);
    set_er_range(1.0f, 0.5f);
    set_df_range(0.1f, 1.0f);
    set_decay(DecayType::expon, max_episodes);
}

/* */
void ReinfLearn::set_decay(DecayType _decay, int nb_episodes) {
    decay = _decay;
    initialise_decay(nb_episodes);
}

/* */
void ReinfLearn::initialise_decay(int nb_episodes) {
    if (nb_episodes == 0) throw runtime_error("Number of episodes is zero");
    if (decay == DecayType::linear) {
        lr_decay = (lr_I - lr_F) / nb_episodes;
        er_decay = (er_I - er_F) / nb_episodes;
        df_decay = (df_I - df_F) / nb_episodes;
    }
    else {
        lr_decay = (lr_I / lr_F - 1.0f) * (1.0f / nb_episodes);
        er_decay = (er_I / er_F - 1.0f) * (1.0f / nb_episodes);
        df_decay = (df_I / df_F - 1.0f) * (1.0f / nb_episodes);
    }
}

/* */
ReinfLearn::~ReinfLearn() {
    if (debug != nullptr) delete debug; 
}

SystemStateInfo* ReinfLearn::get_state_info(const std::string& key) {
    auto it = states.find(key);
    if (it != states.end()) return &(*it).second;
    return NULL;
}

void ReinfLearnAgent::update_trajectory() {
    string a = rl.system.hash_action(*current_action);
    string s = rl.system.hash_state(*previous_state);
    string s_ = rl.system.hash_state(*current_state);
    Transition transition(s, s_, a, 0);
    trajectory.transitions.push_back(transition);
    
    SystemStateInfo *si = rl.get_state_info(s_); // Assumed state s_ is registered
    if(si) si->depth++;
}

/* Apply current policy. false => evaluate only */
/// Pre-requisite: "::reset" applied  
void ReinfLearn::apply_policy(ReinfLearnAgent &_agent, bool learn) {
    while (agent.step()) { // Take actions until terminal state reached
        register_state(*agent.current_state);// Record new state if not seen before
        float reward = system.compute_reward(*agent.previous_state, *agent.current_state, *agent.current_action);
        agent.current_reward = reward; 
        agent.update_trajectory(); // Record the latest state change info
       
        /// Revise q-table
        if (learn)
            update_Q(*agent.previous_state, *agent.current_state, *agent.current_action, agent.current_reward);
    }
}

/* Iterate through trajectory and learn */
void ReinfLearn::propagate(Trajectory& trajectory) {
    for (auto& sci : trajectory.transitions) {
        int a = system.get_action_index(sci.hash_action);
        
        SystemStateInfo* state = get_state_info(sci.hash_state);
        if (!state) { *state = register_state(sci.hash_state); }

        SystemStateInfo* next_state = get_state_info(sci.hash_next_state);
        if (!next_state) { *next_state = register_state(sci.hash_state); }

        updateQ(*state, *next_state, a, sci.reward);
    
    }
}

/* Iterate through trajectory (in reverse) and learn */
void ReinfLearn::back_propagate(Trajectory &trajectory) {
    float G = 0; 
    /*for (auto& sci : trajectory.sci | std::views::reverse) {
        G = agent.df * G + sci.reward; // accumulate rewards     
        int a = system.get_action_index(sci.hash_action);
        SystemStateInfo* state = get_state_info(sci.hash_state);
        if (state == nullptr) state = register_state(sci.hash_state); 
        state->count[a]++;            
        state->q[a] = state->q[a] + agent.lr * (G - state->q[a]);
    }*/

    for (auto& transition : trajectory.transitions | std::views::reverse) {
        int a = system.get_action_index(transition.hash_action);

        SystemStateInfo* state = get_state_info(transition.hash_state);
        if (!state) { *state = register_state(transition.hash_state); }

        SystemStateInfo* next_state = get_state_info(transition.hash_next_state);
        if (!next_state) { *next_state = register_state(transition.hash_state); }

        updateQ(*state, *next_state, a, transition.reward);
    }
}

/* Extract status information about a specific action  */
Point ReinfLearn::get_status(SystemState* state, Action action) {
   pair<string, int> key = get_key(*state, action);
    SystemStateInfo* details = get_state_info(key.first);
    int a = key.second;
    if (stoch)
        return Point{ details->qmean[a], -details->qvar[a] };
    else
        return Point{ details->q[a], -(float) details->count[a] };
}

/* */
pair<string, int> ReinfLearn::get_key(SystemState& state, Action& action) {
    int a = system.get_action_index(action);
    string id = system.hash_state(state);
    return pair<string, int>(id, a);
}

/* Return a particular Q value */
float ReinfLearn::get_Q(SystemState& state, Action& action) {
    pair<string,int> key = get_key(state, action);
    SystemStateInfo* details = get_state_info(key.first);
    int a = key.second;
    return (stoch) ? details->qmean[a] : details->q[a];
}

/* Iterate through the edges of node state_id */
float ReinfLearn::get_max_Q(const std::string& hash_state) {
    SystemStateInfo* si = get_state_info(hash_state);
    return get_max_Q(*si);
}

float ReinfLearn::get_max_Q(SystemStateInfo &state) {
    const auto& values = stoch ? state.qmean : state.q;
    float max_Q = -infinity;   
    for (int a = 0; a < (int) values.size(); a++) {
        if (state.count[a] > 0)
            max_Q = fmax(max_Q, values[a]);
    }
    return (max_Q == -infinity) ? 0.0f : max_Q;
}

/* Update centrally controlled Q-table */
/* The learning_rate and discount is from the agent calling this function */

void ReinfLearn::update_Q(SystemState& previous, SystemState& next, Action& action, float reward) {
    int a = system.get_action_index(action);    
    string s = system.hash_state(previous);
    string s_ = system.hash_state(next);    
    SystemStateInfo* si_s = get_state_info(s);
    SystemStateInfo* si_s_ = get_state_info(s_);
    updateQ(*si_s, *si_s_, a, reward);
}

void ReinfLearn::updateQ(SystemStateInfo& state, SystemStateInfo& next_state, int a, float reward) {
    (this->*updateQfn)(state, next_state, a, reward);
}

void ReinfLearn::updateQ_det(SystemStateInfo& state, SystemStateInfo& next_state, int a, float reward) {
    float max_q = get_max_Q(next_state);
    float new_q = reward + agent.df * max_q;
    state.q[a] = (1 - agent.lr) * state.q[a] + agent.lr * new_q;
    state.count[a]++;        
}

void ReinfLearn::updateQ_stoch(SystemStateInfo &state, SystemStateInfo &next_state, int a, float reward) {
    state.count[a]++; 
    float new_q = reward + agent.df * get_max_Q(next_state);
    float prev_mean = state.qmean[a];
    SimTime delta = (new_q - prev_mean);
    state.qmean[a] += delta / state.count[a];
    state.qvar[a] = ((state.count[a] - 1) * state.qvar[a] + delta * (new_q - state.qmean[a])) / (state.count[a] - 1);
}

void ReinfLearn::set_lr_range(float lr_i, float lr_f) { lr_I = lr_i; lr_F = lr_f; }
void ReinfLearn::set_er_range(float er_i, float er_f) { er_I = er_i; er_F = er_f; }
void ReinfLearn::set_df_range(float d_i, float d_f) { df_I = d_i; df_F = d_f; }

/* Determine the earliest possible future time at which actions can be taken */
optional<SimTime> ReinfLearn::go_to_next_decision_point(SystemState& state) {
    bool exclude_current = false; // Exclude current decision point
    Actions actions;     
    while (auto next_point = system.get_next_decision_point(state, exclude_current)) {                          
        system.transition_to_time(state, *next_point);// Perform state transition                     
        system.get_available_actions(state, actions); // Get actions achievable for the current state
        if (actions.not_empty()) return next_point;// Exit and return the next decision point 
        exclude_current = true; // Bypass current time point                          
    }
    return nullopt;
}


// _________________________________________________________________________
// PRUNE BY COUNT
// -------------------------------------------------------------------------

/* Remove a state from the unordered map and update terminal_states if necessary */
void ReinfLearn::remove_state(const string& state) {  
    states.erase(state); 
    terminal_states.erase(state);
}

/* Prune the "n" oldest states */
void ReinfLearn::prune_oldest_states(int n) {
    multimap<int, string> age_to_id; // States can share the same age
    for (const auto& [id, state] : states) 
        age_to_id.insert({episode - state.created, id}); // (age, id)
    int count = 0;
    for (const auto& [age, id] : age_to_id) {
        if (count++ == n) break;
        remove_state(id);
    }
}

/* Prune the "n" least visited states */
void ReinfLearn::prune_least_visited_states(int n) {
    multimap<int, string> _map;
    for (auto& [key, state] : states) {
        _map.insert({ state.visited, key });
    }
    int count = 0;
    for (const auto& [age, id] : _map) {
        if (count++ == n) break;
        remove_state(id);
    }
}

/* Prune the "n" most dormant states, i.e., lest recently used */
void ReinfLearn::prune_most_dormant_states(int n) {
    multimap<int, string> _map;
    for (auto& [id, state] : states) {
        int dormant = episode - state.last_access;
        _map.insert({ dormant, id });
    }
    int count = 0;
    for (const auto& [dormant, id] : _map) {
        if (count++ == n) break;
        remove_state(id);
    }
}

/* Prune the "n" worst states */
// SLOW!
void ReinfLearn::prune_lowest_value_states(int n) {
    /*multimap<float, string> _map;
    for (auto& [id, state] : states) {
        float score = -infinity; 
        for (int a = 0; a < state.q.size(); a++) score = fmax(score, state.q[a]);
        _map.insert({ score, id });
    }
    int count = 0;
    for (const auto& [score, id] : _map) {
        if (++count == n) break;
        remove_state(id); 
    }*/
    // OR:
    // Store only the n worst states in the priority_queue
    using Pair = std::pair<float, std::string>; // {score, id}
    auto compare = [](const Pair& a, const Pair& b) { return a.first < b.first; };
    std::priority_queue<Pair, vector<Pair>, decltype(compare)> max_heap(compare);
    for (auto& [id, state] : states) {
        float score = -std::numeric_limits<float>::infinity();
        for (float qval : state.q) score = fmax(score, qval);
        if (max_heap.size() < n) max_heap.push({ score, id }); // There is room for more states
        else if (score < max_heap.top().first) { // Has worse performance, so keep it
            max_heap.pop(); // Remove current head
            max_heap.push({ score, id });
        }
    }

    // Remove the stored states
    std::vector<std::string> to_remove;
    while (!max_heap.empty()) {
        to_remove.push_back(max_heap.top().second);
        max_heap.pop();
    }
    for (auto& id : to_remove) remove_state(id);
}

void ReinfLearn::prune_random(int n) {
    std::vector<string> keys;
    keys.reserve(states.size());
    for (auto& [key, _] : states) keys.push_back(key);
    std::shuffle(keys.begin(), keys.end(), gen); // Reorder randomly
    keys.resize(n); // Keep the first n
    for (auto& key : keys) remove_state(key);
}

// _________________________________________________________________________
// PRUNE BY THRESHOLD
// -------------------------------------------------------------------------

/* Prune states by age threshold */
void ReinfLearn::prune_old_states(int max_age) {
    for (auto it = states.begin(); it != states.end(); ) {
        int age = episode - it->second.created;
        if (age > max_age)
            it = states.erase(it); // erase returns the next valid iterator
        else
            ++it;
    }
}
/* Prune states that have been dormant too long */
void ReinfLearn::prune_dormant_states(int max_dormant) {
    for (auto it = states.begin(); it != states.end(); ) {
        int dormant = episode - it->second.last_access;
        if (dormant > max_dormant)
            it = states.erase(it); // erase returns the next valid iterator
        else
            ++it;
    }
}

/* Prune all states worse than the given threshold */
void ReinfLearn::prune_low_value_states(float threshold) {
    for (auto it = states.begin(); it != states.end(); ) {
        float score = 0; // ?
        if (score > threshold)
            it = states.erase(it); // erase returns the next valid iterator
        else
            ++it;
    }
}

// _________________________________________________________________________
// CULLING STRATEGIES
// -------------------------------------------------------------------------

/* Direct enforcement of memory */
void ReinfLearn::cull_states_direct(int option, int episode) {
    long nb_states = (long) states.size();
    bool too_many = (nb_states > max_states);
   
    if (episode % prune_interval == 0 and too_many) {
        int nb_cull = nb_states - max_states;
        cout << " -> Culled " << nb_cull << " states";
        switch (option) {
        case 1: prune_least_visited_states(nb_cull); break;
        case 2: prune_most_dormant_states(nb_cull); break;        
        case 3: prune_oldest_states(nb_cull); break;
        case 4: prune_lowest_value_states(nb_cull); break;    
        case 5: prune_random(nb_cull); break;
        }
    }
}

void ReinfLearn::cull_states_indirect() {
    for (auto it = states.begin(); it != states.end(); ) {
        int age = episode - it->second.created;
        int dormant = episode - it->second.last_access;
        if (dormant > max_dormant or age > max_age)
            it = states.erase(it); // erase returns the next valid iterator
        else
            ++it;
    }
}

/* Try to emplace if not already present */
/* The SystemStateInfo object is created conditionlly */
SystemStateInfo& ReinfLearn::register_state(const string& key) {
    int nb_actions = system.number_of_actions();
    auto [it, inserted] = states.emplace(key, SystemStateInfo(key, nb_actions, stoch));
    if (!inserted) it->second.last_access = episode;
    it->second.visited++; 
    return it->second;
}

SystemStateInfo& ReinfLearn::register_state(SystemState& state) {
    string hash_state = system.hash_state(state);   
    return register_state(hash_state);   
}

/* Take an action, identify the next decision point 
   after the action and transition system to the new state */
void ReinfLearn::teach_agent(int phase, int nb_episodes) {
    for (episode = 0; episode < nb_episodes; episode++) { 
        system.reset(); // Reset system to state_zero  
        agent.reset(*system.state_zero); // Reset agent hyperparameters
        apply_policy(agent, true); // Take full set of actions and update the q-values        
        float score = evaluate_actions(agent.actions, true);
        agent.update_parameters();
        float per = 100 * (float) (episode+1) / (float)nb_episodes;
        cout << "\r\33[2KRunning RL -> " << per << "%";
        // ___________________________________________________
        /// Prune at the end of each episode
        //int option = 1; // 1 => least_visited_states; 2 => most_dormant; 3 => oldest
        //cull_states_indirect(); // Cull by age and dormancy
        //cull_states_direct(5, episode); // Cull to reduce memory
    }
}

/* Evaluate a set of actions and update best */
// Assumption: Only if a full set of actions is passed should flag = true 
// Assumption: The system is in a given initial state for this evaluation
// i.e., actions are not made from another state
float ReinfLearn::evaluate_actions(vector<Action>& actions, bool flag) {
    float score = system.evaluate_actions(actions, stoch);
    if(flag) update_best(actions, score);  
    return score;
}

/* */
void ReinfLearn::update_terminal_states(SystemState& state) {
    string key = system.hash_state(state); // Assumed to be terminal
    if (terminal_states.find(key) == terminal_states.end()) terminal_states.insert(key);
}

/* Standard QL. However, it is setup to run multiple phases of learning */
/* IMPORTANT NOTE: MUST RESET "best_score" BEFORE RUNNING! */
void ReinfLearn::run_ql(int nb_phases, int episodes_per_phase) {
    int max_episodes = nb_phases * episodes_per_phase;
    set_decay(DecayType::expon, episodes_per_phase);
    prepare_agent(mode, decay);
    wall_start = omp_get_wtime();
    for (int phase = 0; phase < nb_phases; phase++) {       
        for (episode = 0; episode < episodes_per_phase; episode++) {
            system.reset(); // Reset system to state_zero  
            agent.reset(*system.state_zero); // Reset agent hyperparameters
            apply_policy(agent, true);  // Take a sequence of actions using current knowledge            
            float score = evaluate_actions(agent.actions, true);
            agent.update_parameters();
            float per = 100 * (float)(episode + 1) / (float)episodes_per_phase;
            cout << "\r\33[2KRunning RL -> " << per << "%";
        }
        agent.reset_parameters();

        // Optional - Pruning at the end of each phase
        //cull_states_indirect(); // Cull by age and dormancy
        //cull_states_direct(5); // Cull to reduce memory
    }
    wall_end = omp_get_wtime();
    wall_time = wall_end - wall_start;
}

/* Monte Carlo QL - incremental, no returns */
void ReinfLearn::run_mc_ql(int episodes) {

    wall_start = omp_get_wtime();

    /// 1. Initialise parameters and objects
    set_decay(DecayType::linear, episodes);
    prepare_agent(mode, decay); // Set agent hyper-parameters  

    /// 2. Start
    wall_start = omp_get_wtime(); // Note start time
    for (episode = 0; episode < episodes; episode++) {
        cout << "\nEpisode " << episode << ". LR = " << agent.lr << ". Dis: " << agent.df;
        system.reset(); // Reset system to state_zero 
        agent.reset(*system.state_zero);  // Set agents previous and current state
        apply_policy(agent, false);// Learn = false
        back_propagate(agent.trajectory);
        float score = evaluate_actions(agent.actions, true);
        agent.update_parameters();
    }
    wall_end = omp_get_wtime();
    wall_time = wall_end - wall_start;
}

/* */
void ReinfLearn::run_sims(bool make) {
    for (int j = 0; j < 100; j++) {
        system.reset(); // Reset system to state_zero 
        agent.reset(*system.state_zero);  
        cout << "\n";
        SimResult result;
        system.simulate(result);
        if (make) {
            agent.trajectory = system.make_trajectory(*system.state_zero, result.actions_log, result.sim_end);
            system.report_trajectory(cout, agent.trajectory);
        }
        result.actions_log.clear();            
    }
}

// A(s,a) = Q(s,a) - V(s)
// A > 0 => action is better than average
// A < 0 => action is worse than average
float ReinfLearn::compute_advantage(SystemState& state, Action& action) {
    SystemStateInfo& si = register_state(state);
    //float adv = si.value; 
    float adv = get_Q(state, action) - si.value;
    return adv;
}

/* Apply value function. Set V(s) = value  */
void ReinfLearn::update_value(SystemState& state, float value) {
    SystemStateInfo& si = register_state(state);
    si.visited++;
    //si.value += (value - si.value) / si.visited;   // Stochastic case

    si.value = max(si.value, value); // Greedy approach
}

/* */
SystemState ReinfLearn::choose_starting_state(float prob) {    
    /// No options other than state_zero
    int sz = static_cast<int>(pool.size());
    if (sz == 0) return *system.state_zero; 
    
    /// Choose state_zero by probability
    std::bernoulli_distribution choose(prob); 
    if (choose(system.gen)) return *system.state_zero;

    /// Choose according to interest level
    std::vector<double> weights;
    weights.reserve(sz);   
    
    std::vector<std::string> keys;
    keys.reserve(sz);
        
    for (const auto& key : pool) {
        keys.push_back(key);
        auto it = states.find(key);
        double w = (it != states.end()) ? it->second.int_lev : 0.0;
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
    
    return system.unhash_state(keys[index]);
}

/* Update pool of interesting states */
void ReinfLearn::add_to_pool(SystemStateInfo& state) {    

    /// Already in pool -> do nothing
    if (pool.find(state.key) != pool.end()) return;

    /// Space available
    if (pool.size() < max_pool_size) {
        pool.insert(state.key);
        pool_ordered.insert({ state.int_lev, state.key });
        return;
    }
 
    /// Space not available - replace worst
    auto lowest = *pool_ordered.begin(); // Note: highest = *pool_ordered.rbegin();
    if (state.int_lev > lowest.first) {
        pool.erase(lowest.second);
        pool_ordered.erase(pool_ordered.begin());
        pool.insert(state.key);
        pool_ordered.insert({ state.int_lev, state.key });        
    }
}

/* */
void ReinfLearn::update_value(string name, float new_value) {
    SystemStateInfo* info = get_state_info(name);
    info->value = max(info->value, new_value); 
}


/* Update value parameter of each state in the vector */
void ReinfLearn::update_values(Trajectory &trajectory) {
    vector<Transition>& trans = trajectory.transitions;
    float G = 0.0f, gamma = 0.9f;
    for (int h = (int) trans.size() - 1; h >= 0; --h) {
        G = trans[h].reward + gamma * G; // future reward   
        update_value(trans[h].hash_state, G);
    }
}

/* */
void ReinfLearn::register_trajectory(Trajectory& trajectory) {
    vector<Transition>& transitions = trajectory.transitions; 
    SystemState s = system.unhash_state(transitions[0].hash_state);
    register_state(s);
    for (int h = 0; h < (int) transitions.size(); h++) {
        SystemState s_ = system.unhash_state(transitions[h].hash_next_state);
        register_state(s_);
    }
}

Action ReinfLearn::get_first_action(Trajectory& trajectory) {
    Transition& trans = trajectory.transitions.front();
    Action action = system.unhash_action(trans.hash_action);
    return action;
}


/* Output Q results*/
void ReinfLearn::report_Q_table(const std::string& file) {
    ofstream out(file);
    int i_ = 0;
    for (auto& state : states) {        
        out << "\n[" << i_ << "] ";
        out << '"' << state.first.c_str() << '"';
        int n = (int) state.second.count.size();
        out << "---> ";
        if (stoch) {
            for (int i = 0; i < n; i++) 
                if (state.second.count[i] > 0)
                    out << i << " (" << state.second.count[i] << ", " << state.second.qmean[i] << "), ";
        }
        else {
            for (int i = 0; i < n; i++) 
                if (state.second.count[i] > 0)
                    out << i <<" (" << state.second.count[i] << ", " << state.second.q[i] << "), ";
        }
        i_++;
    }    
}

void ReinfLearn::report_terminal_states(const string& file) {
    ofstream out(file);
    for (auto state : terminal_states) out << "\n" << state.c_str();
}

/* Set agents learning rate, exploration mode/rate, discount factor */
void ReinfLearn::prepare_agent(ExplorationMode _mode, DecayType _decay_mode) {
    agent.mode = mode;
    agent.decay = _decay_mode;
    agent.set_parameters(lr_I, lr_decay, er_I, er_decay, df_I, df_decay);
}
   
/* */
void ReinfLearn::report(ostream &out, int episodes) {
    out << "\n#Episodes: " << episodes;
    out << "\n#States = " << states.size();
    out << "\n#TerminalStates = " << terminal_states.size();
    out << "\nLearning rate: " << lr_I << "->" << lr_F;
    out << "\nExploration rate: " << er_I << "->" << er_F;
    out << "\nDiscount: " << df_I << "->" << df_F;
    out << "\nBest score: " << best_score;
    out << "\nWall time: " << wall_time;
   
    for (auto& action : best_actions) 
        out << "\n" << system.hash_action(action) << ",";
}

/* */
void ReinfLearn::delete_actions(vector<Action*>& actions) { 
    for (auto& action : actions) delete action; 
}

/* Largest */
void ReinfLearn::update_best(vector<Action> &actions, float score) {    
    if (score > best_score) {
        best_actions = actions; // Overwrite
        best_score = score;
    }
}

/* Convert list of strings to a list of Actions */
vector<Action> ReinfLearn::convert(vector<string>& actions_code) {
     vector<Action> actions;
     for (string code : actions_code) {
        int a = system.find_action(code);
        Action action = system.get_action(a);
        actions.push_back(action);
     }
     return actions;
}

/* Experimental. Works fine but does not find best solution */
void ReinfLearn::extract_optimal_policy() {
    system.reset(); // Reset system to state_zero    
    SystemState* state = system.state_zero;
    Actions actions;
    int iter = 0;
    cout << "\nBest Policy: ";
    while(!system.is_terminal_state(*state))
    {      
        Action action = get_best_action(*state);
        system.report_action(cout, action);
        cout << "@ t= " << state->time << "; Completes @ t= " << state->time + 0;
        system.apply_action(*state, action);
        
        bool exclude_current = false;
        while (true) {
            auto next_point = system.get_next_decision_point(*state, exclude_current);
            if (next_point == nullopt)  break;            
            system.transition_to_time(*state, *next_point);
            system.get_available_actions(*state, actions);
            if (actions.size() > 0) break;
            exclude_current = true;
        }              
        iter++;
    } 
    cout << "\nFinal time = " << state->time;
}

void ReinfLearn::get_memory() {
    cout << "\nSystemStateInfo: ";
    get_mem_used(states);
}

/*********************************************************/
/* ------------------- MEMORY USAGE -------------------- */
/*********************************************************/

size_t get_mem_used(const unordered_map<string, SystemStateInfo> &states) {
    size_t total = sizeof(states);
    size_t tot_arr = 0;
    size_t tot_edges = 0;
    size_t tot_states = 0;
    for (const auto& [key, details] : states) {
        total += sizeof(key);    // string object overhead
        total += key.capacity(); // string contents (char array)
        total += sizeof(details);   // vector object overhead

        tot_arr = 3 * details.q.capacity() * sizeof(float);
        tot_arr += details.count.capacity() * sizeof(int);
        
        //tot_edges += sizeof(details.edges);
        //tot_edges += details.edges.size() * sizeof(std::tuple<string, string, float>);
        //tot_states += get_mem_used(details.info); // SystemState
    }
    total += tot_arr;
    total += tot_edges;
    total += tot_states;
    cout << "\nTot Arr = " << to_gigabytes(tot_arr) << "GB";
    cout << "\nTot Edges = " << to_gigabytes(tot_edges) << "GB";
    cout << "\nTot States = " << to_gigabytes(tot_states) << "GB";
    cout << "\nTotal = " << to_gigabytes(total) << "GB";
    return total;
}



/*********************************************************/
/* ------ SIMULATION ASSISTED GREEDY SEARCH  CLASS ----- */
/*********************************************************/

/* */
SAGS::SAGS(System& _system, std::mt19937& _gen):system(_system), gen(_gen) {
    previous_state = new SystemState();
    current_state = new SystemState();
}

/* */
SAGS::~SAGS() {
    delete previous_state;
    delete current_state;
}

/* */
float SAGS::evaluate_actions(vector<Action>& actions, bool flag) {
    float score = system.evaluate_actions(actions, false);
    if (flag) update_best(actions, score);
    return score;
}

/* */
void SAGS::update_best(vector<Action>& actions, float score) {
    if (score > best_score) {
        best_actions = actions; // Overwrite
        best_score = score;
    }
}

/* */
void SAGS::report(ostream& out, int episodes) {
    out << "\n#Episodes: " << episodes;
    out << "\n#TerminalStates = " << terminal_states.size();
    out << "\nBest score: " << best_score;
    out << "\nWall time: " << wall_time;

    for (auto& action : best_actions)
        out << "\n" << system.hash_action(action) << ",";
}

/* */
bool SAGS::take_action(Action& action) {
    action.when = current_state->time;
    current_action = action; 
    previous_state->copy(*current_state);
    auto next_point = system.transition(*current_state, *current_action);
    return (next_point != nullopt);
}

/* */
void SAGS::update_terminal_states(SystemState& state) {
    string key = system.hash_state(state); // Assumed to be terminal
    if (terminal_states.find(key) == terminal_states.end()) terminal_states.insert(key);
}

void SAGS::reset(SystemState& starting) {
    actions.clear();    
    current_state->copy(starting); // Copy the starting state
    previous_state->records.clear();
    previous_state->is_terminal = false;
    previous_state->time = 0;
}

Action SAGS::lookahead_parallel(SystemState& current, int nsim, std::optional<int> steps, int nthrds) {
    vector<Trajectory> trajectories = system.batch_simulate(nsim, nthrds, *steps, current);
    float value, best_value = -infinity;   
    int best_at = 0;    
    for (int i = 0; i < nsim; i++) {
        trajectories[i].value = get_discounted_reward(trajectories[i].transitions);
        value = trajectories[i].value;
        if (value > best_value) {
            best_value = value;           
            best_at = i;
        }   
    }    

    /// Extension: could call update_best here - maybe the trajectory
    /// is better than any other observed trajectory    

    Transition& trans = trajectories[best_at].transitions.front();
    Action action = system.unhash_action(trans.hash_action);
    return action;
}

/* */
Action SAGS::lookahead(SystemState& current, int nsim, std::optional<int> steps) {    
    float best_value = -infinity;
    Action action;
    for (int i = 0; i < nsim; i++) {
        Trajectory trajectory = get_simulated_trajectory(system, current, steps);

        /// Extension: could call update_best here - maybe the trajectory
        /// is better than any other observed trajectory 
       
        float future_value = trajectory.value;        
        if (future_value > best_value) {
            best_value = future_value;           
            Transition& trans = trajectory.transitions.front();
            action = system.unhash_action(trans.hash_action);
        }
    }
    return action;
}

/* */
Action SAGS::lookahead_choose_action(SystemState& current, int nsim, std::optional<int> steps, std::optional<int> nthrds){    
    if (parallel) 
        return lookahead_parallel(current, nsim, steps, *nthrds);
    else 
        return lookahead(*current_state, nsim, steps);
}

/* */
Action SAGS::choose_random_action(SystemState &state, bool rand) {
    Actions _actions;
    system.get_available_actions(state, _actions);
        
    if (rand) {    
        std::uniform_int_distribution<> unif(0, static_cast<int>(_actions.info.size()) - 1);
        Action action = _actions.info[unif(gen)];
        return action;
    }
    else {
        vector<float> weights;
        float value;
        for (auto& action : _actions.info) {
            SystemState _state = system.get_next_state(action, state);
            
            string key = system.hash_state(_state);
            SystemStateInfo* info = get_state_info(key);
            // info->value may be negative, so scale            
            value = (info == nullptr) ? 0.001f : max_float - info->value;
            weights.push_back(value);
        }
        std::discrete_distribution<> dist(weights.begin(), weights.end());
        Action action = _actions.info[dist(gen)];
        return action;
    }
}

/* */
void SAGS::run(int episodes, int nsim, std::optional<int> steps, float epsilon, std::optional<int> nthrds, bool learn) {
    parallel = (nthrds) ? true: false; 
    if (parallel) system.setup_components(*nthrds, system.gen());

    std::uniform_real_distribution<> distr(0, 1);
    best_score = -infinity;
    wall_start = omp_get_wtime();   
    for (int i = 0; i < episodes; i++) {

        SystemState* initial_state = system.state_zero; // OR: choose_starting_state();
        system.restore_state(*initial_state);
        reset(*initial_state); // Re-initialise current state, previous state, action list

        int step = 0;
        while (!current_state->is_terminal) {
            step++; 
            Action action;
            float rv = (float)distr(gen);

            /// CHOOSE RANDOM ACTION OR VIA LOOK-AHEAD (USING SIMULATION)
            if (rv < epsilon) action = choose_random_action(*current_state, learn);                        
            else action = lookahead_choose_action(*current_state, nsim, steps, nthrds);                                      

            system.restore_state(*current_state);
            take_action(action);  // Update the system          
            actions.push_back(action); // Append to the sequence of actions
        }

        /// TERMINAL STATE REACHED
        update_terminal_states(*current_state);

        /// MAKE AND REGISTER FINAL TRAJECTORY
        if (learn) {            
            vector<pair<int, SimTime> > _vector;
            for (auto& action : actions) { 
                int a = system.get_action_index(action);
                _vector.push_back(pair<int, SimTime>(a, action.when));
            }
            Trajectory _trajectory = system.make_trajectory(*initial_state, _vector, current_state->time);
            record(_trajectory);            
        }

        /// EVALUATION       
        float score = evaluate_actions(actions, true); // true => Update best_score         

        /// REPORT
        cout << "\nEpisode " << i << " -> " << score << " / best = " << best_score;   
    }

    wall_end = omp_get_wtime();
    wall_time = wall_end - wall_start;

    if (parallel) system.delete_components();
}

/* */
SystemStateInfo* SAGS::get_state_info(const std::string& key) {
    auto it = states.find(key);
    if (it != states.end()) return &(*it).second;
    return NULL;
}

/* */
SystemStateInfo& SAGS::register_state(const string& key) {
    int nb_actions = system.number_of_actions();
    auto [it, inserted] = states.emplace(key, SystemStateInfo(key, nb_actions, false));   
    it->second.visited++;
    return it->second;
}

/* */
SystemStateInfo& SAGS::register_state(SystemState& state) {
    string hash_state = system.hash_state(state);
    return register_state(hash_state);
}

/* */
void SAGS::update_value(string name, float new_value) {
    SystemStateInfo* info = get_state_info(name);
    info->value = max(info->value, new_value);
}

/* */
void SAGS::record(Trajectory& trajectory) {
    vector<Transition>& trans = trajectory.transitions;
    SystemState s = system.unhash_state(trans[0].hash_state);
    register_state(s);
    float G = 0.0f, gamma = 0.9f;
    for (int h = 0; h < (int)trans.size(); h++) {
        SystemState s_ = system.unhash_state(trans[h].hash_next_state);
        register_state(s_);
        G = trans[h].reward + gamma * G; // the future reward
        update_value(trans[h].hash_state, G);
    }
}


