#pragma once

/*********************************************************/
/* -------------------- LIBRARIES  --------------------- */
/*********************************************************/

#include <ranges>
#include <map>
#include <omp.h>        // OpenMP parallel processing

#include "simulator.h"

/*********************************************************/
/* -------------------- DEFINITIONS -------------------- */
/*********************************************************/

using namespace std;

typedef vector<float> Point;

enum class ExplorationMode { e_greedy, pareto };
enum class DecayType { linear, expon };

/*********************************************************/
/* --------------- FORWARD DECLARATRIONS --------------- */
/*********************************************************/

class ReinfLearn;
class ReinfLearnAgent;
class SystemStateInfo;

class Agent;
class SimBasedRL;

/*********************************************************/
/* --------- REINFORCEMENT LEARNING FUNCTIONS ---------- */
/*********************************************************/

float get_discounted_reward(vector<Transition>& trans);
Trajectory get_simulated_trajectory(System& system, SystemState& state, optional<int> steps);
vector<Trajectory> batch_simulate(int nsim, int nthrds, System& system, SystemState& state, optional<int> steps);

/*********************************************************/
/* --------- REINFORCEMENT LEARNING AGENT CLASS -------- */
/*********************************************************/

class ReinfLearnAgent {
public:
	using UpdateFunction = void (ReinfLearnAgent::*)();

	ReinfLearn& rl;	
	
	/* Settings */
	float lr,lr_I,lr_decay;  // Learning rate
	float er,er_I,er_decay;  // Exploration rate
	float df,df_I,df_decay;  // Discount factor
	ExplorationMode mode;
	DecayType decay;
	UpdateFunction update_fn;
	
	/* Attributes */
	float current_reward = 0;
	SystemState *current_state, *previous_state;
	optional<Action> current_action;
	Trajectory trajectory;
	vector<Action> actions;
	
	/* Functions */
	ReinfLearnAgent(ReinfLearn &rl);
	~ReinfLearnAgent();
	void set_parameters(float _lr_I, float _lr_decay, float _er_I, float _er_decay, float _df_I, float _df_decay);
	void reset_parameters();
	bool step();
	bool take_action(Action& action);
	void update_system(Action& action, SystemState& state);
	void revise_linear();
	void revise_expon();
	void update_parameters();
	void update_trajectory();
	Action select_action(Actions& actions);
	Action epsilon_greedy(Actions& actions);
	Action pareto_selection(Actions& actions);
	void reset(SystemState &state);
	float compute_accumulated_reward();
};

/*********************************************************/
/* ---------------- COMPARISON CLASS ------------------- */
/*********************************************************/

// Each string represents a hashed system state
// Ordering is based upon the state time, that is placed 
// at the start of each string
struct ChronOrder {
	bool operator()(const std::string& a, const std::string& b) const {		
		vector<string> sub_str_a = split_string(a, ',');
		vector<string> sub_str_b = split_string(b, ',');
		SimTime t_a = (SimTime) atoi(sub_str_a[0].c_str());
		SimTime t_b = (SimTime) atoi(sub_str_b[0].c_str());
		if (t_a != t_b) return (t_a < t_b);
		else return (a < b);
	}
};

/* Custom comparator for a <state, action, reward> tuple */
struct compare_info {

	//bool operator()(const tuple<string, string, float>& p1, const tuple < string, string, float>& p2) const {
	bool operator()(const tuple<string, string>&p1, const tuple<string, string>&p2) const {
		// First, compare the first element of the pair
		if (get<0>(p1) != get<0>(p2)) {
			return get<0>(p1) < get<0>(p2);
		}
		// If the first elements are the same, compare the second element of the pair
		return get<1>(p1) < get<1>(p2);
	}
};

/*********************************************************/
/* ----------------  STATE INFO CLASS ------------------ */
/*********************************************************/

class SystemStateInfo {
public:
	string key; // Hashed state
	
	/* Attributes */
	vector<float> q, qmean, qvar; // QTable: One element for each action
	vector<int> count; // QTable: How many times each action has been taken
	vector<float> prob; // Policy - probability
	vector<float> theta; // Policy - raw score
	bool is_terminal = false;

	int depth; // Depth (a.k.a. position) within state transition tree
	int created, last_access, visited;
	float value = -infinity; 
	float int_lev; // interest level 
	
	/* Functions */
	SystemStateInfo(string _key, int nb_actions, bool stoch = false);
	~SystemStateInfo();
	void initialise_arrays(int nb_actions, bool stoch);
};

/*********************************************************/
/* ----------- REINFORCEMENT LEARNING CLASS ------------ */
/*********************************************************/

class ReinfLearn {
public:
	using UpdateQFn = void (ReinfLearn::*)(SystemStateInfo& details, SystemStateInfo& details_, int a, float reward);

	ReinfLearnAgent agent; 
	ostream* debug;
	
	/* References */
	std::mt19937& gen; // Random number generator
	System& system; // Reference to some system

	/* State Attributes */	
	set<string> terminal_states;
	unordered_map<string, SystemStateInfo> states; // Dictionary
	vector<Action> best_actions; 
	float best_score = -infinity;	
	long state_counter;	
	double wall_start, wall_end, wall_time;
	int episode;
	
	/* State Pool */	
	set<string> pool;// A collection of interesting states
	set<pair<float, string>> pool_ordered;

	/* Settings - Learning */
	float lr_I, lr_F, lr_decay;
	float er_I, er_F, er_decay;
	float df_I, df_F, df_decay;
	long max_states = 10000; // 1000000; // Target number of states to keep in memory
	int max_pool_size = 100;
	DecayType decay;
	ExplorationMode mode;
	UpdateQFn updateQfn; // Member function
	bool stoch;
	float epsilon_sags = 0.1f;
	int count_roll = 0;
	int count_not_roll = 0;

	/* Settings - Pruning */
	int prune_interval = 200; // After so many episodes, prune
	int max_age = 100;
	int max_dormant = 50;

	/* Functions - Constructor & Destructor */
	ReinfLearn(System& _system, std::mt19937& _gen, bool stoch);
	~ReinfLearn();
	void delete_actions(vector<Action*>& actions);
	
	/* Functions - Reporting */
	void report_Q_table(const std::string& file);
	void report_terminal_states(const string& file);
	void report_transition_tree(string file);
	void report(ostream &os, int episodes);

	/* Functions - Learning Algorithms */
	void run_sims(bool make);
	void run_ql(int nb_phases, int episodes_per_phase);	
	void run_mc_ql(int episodes);
	
	/* Functions - Intitialisations */ 
	void set_lr_range(float lr_i, float lr_f);
	void set_er_range(float er_i, float er_f);
	void set_df_range(float d_i, float d_f);
	void set_default_parameters(int max_episodes);
	void initialise_decay(int nb_episodes);
	void set_decay(DecayType type, int max_episodes);
	void prepare_agent(ExplorationMode _mode, DecayType _decay_mode);

	/* Functions - Updating & Evaluations */
	void update_value(SystemState& state, float value);
	float compute_advantage(SystemState& state, Action& action);
	float evaluate_actions(vector<Action>& actions, bool flag);
	void update_terminal_states(SystemState& state);
	void update_best(vector<Action>& actions, float score);
	optional<SimTime> update_system(SystemState& previous, SystemState& state, Action& action);
	SystemStateInfo& register_state(SystemState& state);
	SystemStateInfo& register_state(const string& hash_state);
	void update_Q(SystemState& previous, SystemState& next, Action& action, float reward);
	void updateQ(SystemStateInfo& details, SystemStateInfo& _details, int a, float reward);
	void updateQ_det(SystemStateInfo& details, SystemStateInfo& _details, int a, float reward);
	void updateQ_stoch(SystemStateInfo& details, SystemStateInfo& _details, int a, float reward);
	
	/* Functions - Access */
	pair<string, int> get_key(SystemState& state, Action& action);
	Point get_status(SystemState* state, Action action);
	Action get_best_action(SystemState &state);
	float get_Q(SystemState& state, Action& action);
	float get_max_Q(const std::string& hash_state);
	float get_max_Q(SystemStateInfo &state);
	
	/* Functions - Misc */
	SystemStateInfo* get_state_info(const std::string& key);
	void apply_policy(ReinfLearnAgent& _agent, bool learn = true);
	void back_propagate(Trajectory& trajectory);
	void propagate(Trajectory& trajectory);
	optional<SimTime> go_to_next_decision_point(SystemState& state);
	void teach_agent(int phase, int nb_episodes);
	void extract_optimal_policy();
	void record_transition(SystemState& previous_state, SystemState& next_state, Action action, float reward);
	void analyse();
	vector<Action> convert(vector<string>& actions_id);
	SystemState choose_starting_state(float prob=0.2);
	void add_to_pool(SystemStateInfo& state);
	void get_memory();
	void update_value(string name, float new_value);
	void update_values(Trajectory& trajectory);
	void register_trajectory(Trajectory& trajectory);
	Action get_first_action(Trajectory& trajectory);

	/* Functions - Memory management */
	void prune_oldest_states(int n);
	void prune_old_states(int max_age);
	void prune_least_visited_states(int n);
	void prune_most_dormant_states(int n);
	void prune_dormant_states(int period);
	void prune_lowest_value_states(int n);
	void prune_low_value_states(float threshold);
	void prune_random(int n);
	void remove_state(const string& state_id);
	void cull_states_direct(int option, int episode);
	void cull_states_indirect();
};

/*********************************************************/
/* ------------------------ SAGS ----------------------- */
/*********************************************************/

class SAGS {
public:
	/* References */
	std::mt19937& gen; // Random number generator
	System& system; // Reference to some system
	bool parallel = false;

	/* Objects */
	set<string> terminal_states;
	vector<Action> best_actions;	
	SystemState* current_state, *previous_state;
	optional<Action> current_action;
	vector<Action> actions;
	unordered_map<string, SystemStateInfo> states; 

	/* Search parameters */
	float best_score = -infinity;
	double wall_start, wall_end, wall_time;
	
	/* Functions */
	SAGS(System& _system, std::mt19937& _gen);
	~SAGS();
	void reset(SystemState& starting_state);
	bool take_action(Action& action);
	float evaluate_actions(vector<Action>& actions, bool flag);
	void update_best(vector<Action>& actions, float score);
	void update_terminal_states(SystemState& state);
	void report(ostream& os, int episodes);
	void run(int episodes, int nsim, std::optional<int> steps, float epsilon, std::optional<int> nthrds, bool learn = false);
	Action lookahead(SystemState &current, int nsim, std::optional<int> ssteps);
	Action lookahead_parallel(SystemState& current, int nsim, std::optional<int> steps, int nthrds);
	Action lookahead_choose_action(SystemState& current, int nsim, std::optional<int> steps, std::optional<int> nthrds);
	Action choose_random_action(SystemState& state, bool rand=true);

	// Learning functionality
	void record(Trajectory& trajectory);
	SystemStateInfo& register_state(const string& key);
	SystemStateInfo& register_state(SystemState& state);
	void update_value(string name, float new_value);
	void update_values(Trajectory& trajectory);
	SystemStateInfo* get_state_info(const std::string& key);
};


/*********************************************************/
/* ------------------- MEMORY USAGE -------------------- */
/*********************************************************/

size_t get_mem_used(const unordered_map<string, SystemStateInfo> &states);


