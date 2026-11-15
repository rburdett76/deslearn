#pragma once

#include "simulator.h"

#include <map>

/*********************************************************/
/* ----------------- ACTOR CRITIC ---------------------- */
/*********************************************************/

// Action Info
class AInfo {
public:
	AInfo(int _a) :a(_a) {}

	Action ref;
	int a;
	int count = 0;
	float prob = 0;
	float theta = 0;	
};

// State Info
class SInfo {
public:
	string key; // Hashed state
	map<int, AInfo> actions;
	
	int visited = 0;
	float value = 0;
	float interest = 0;

	AInfo* add_action(int _a, float _theta, Action _action);
	SInfo(string _key) :key(_key) {};
	~SInfo() = default;
};


class BufferInfo {
public:
	float reward;
	SystemState state;
	SystemState next;
	Action action;
	bool is_terminal;

	BufferInfo(const SystemState& _state, const SystemState& _next, const Action& _action, float _reward, bool _is_terminal)
		:state(_state), next(_next), reward(_reward), action(_action), is_terminal(_is_terminal) {}
};


class ActorCritic {
public:
	/* Referenced Objects */
	System& system;

	/* Runtime */
	double runtime;
	double wall_start, wall_end, wall_time;
	int episode;

	/* Settings */
	bool stoch = false;

	/* Learning Rates */
	// Actor learning rate must usually be smaller than critic, otherwise policy chases noise.
	float alpha_critic = 0.001f; // 0.01f; // Should be faster than actor!
	float alpha_actor = 0.1f * alpha_critic; // 0.005

	/* Gamma */
	// 0.95–0.99 -> standard
	// 0.9 -> more short sighted
	// 0.99–0.995 -> long horizon (common in scheduling)
	float gamma = 0.995f; 

	/* Entropy - Softmax policies can saturate.Eventually one action gets:
	prob ~= 0.99999. Then gradients become tiny and learning stalls.
	Entropy regularization helps enormously.
	*/
	// 0.0001  -> very weak exploration
	// 0.001   -> common starting point
	// 0.01    -> strong exploration
	// 0.1     -> usually too high
	float entropy = 0.001f;
	
	int nreplay = 0;
	int nroll = 10; // [10,50]
	optional<int> rollout_horizon = 5;// [5,50];
	int replay_buffer = 100;
	int trans = 0; // counter for #transitions
	float epsilon, delta, mean_abs_delta = 0;
	bool on_policy = true;
	optional<int> nthrds;

	/* States */
	unordered_map<string, SInfo> states; // Dictionary
	deque<BufferInfo> buffer; // Stored transitions for later replays
	
	vector<Action> best_actions; 
	set<string> terminal_states; 
	float best_score = -infinity; 

	/* State Pool */
	float threshold = 1;
	set<string> pool;// A collection of interesting states
	
	/* Functions */
	ActorCritic(System& _system, bool _stoch);
	~ActorCritic() {}

	bool is_interesting(float delta);
	void record_terminal_state(SystemState& state);
	SInfo* get_info(SystemState& state);
	SInfo* get_info(const std::string& key);
	SInfo& record_state(SystemState& state);
	SInfo& record_state(string &key, SystemState& state);
	SystemState iterate(vector<Action> &actions);
	void add_to_pool(string key, float score);
	void add_to_pool(SystemState& state);
	void update_policy(SystemState& state);
	void actor_update(SystemState& state, Action& action, float delta);
	void update_theta(SystemState& state, Action& action, float delta);
	void critic_update(SystemState& state, float delta);
	void critic_update(string& state, float delta);
	void critic_update(SInfo& info, float  delta);
	void run(int episodes);
	void step(SystemState& state, Action& action, SystemState& next, float& reward);
	float compute_delta(SystemState& state, SystemState& next, float reward);
	float compute_delta(SInfo& state, SInfo& next, float reward, bool next_is_terminal);
	void store_transition(SystemState& state, Action& action, SystemState& next, float reward);
	void random_replay();
	void prioritized_replay();
	void random_replay_rollout();
	float get_discounted_reward(vector<Transition>& trans);
	void process_trajectory(Trajectory& trajectory);
	void sequential_rollouts(SystemState *state);
	void parallel_rollouts(SystemState *state);
	void on_policy_rollout_one(SystemState& state, int steps);
	void off_policy_rollout_one(SystemState& state, int steps);
	void on_policy_rollout(SystemState& state, int H);	
	void off_policy_rollout(SystemState& state, int H);
	void parallel_on_policy_rollouts(SystemState& state, int steps, int nthrds);
	void parallel_off_policy_rollouts(SystemState& state, int steps, int nthrds);
	float on_policy_lookahead(SystemState& state, int nsim, int steps);
	Action sample_action(SystemState& state);
	SystemState choose_state(float prob = 0);
	void update_interest(SystemState& state, SystemState& next, float reward, float delta, int method=1);
	void report(ostream &os);
	void update_best(vector<Action>& actions, float score);
};