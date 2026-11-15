#include "simulator.h"
#include "flex_jobshop.h"
#include "deslearn.h"
#include "actor_critic.h"

#include <iomanip>

#define _CRTDBG_MAP_ALLOC
#include <cstdlib>
#include <crtdbg.h>

int global_seed = 10;
std::mt19937 rng(global_seed); // Global Mersenne Twister engine 


FlexJobShop::JobShop* load_jobshop(string file, Sim *sim) {
	using namespace FlexJobShop;
	JobShop* jobshop = new JobShop(*sim); // Create a jobshop system	
	jobshop->load_instance(file); // Initialise the systems components
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	// Convert to readable formart
	//string name(file + ".fixed");
	//jobshop->save_instance(name); 
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	jobshop->state_zero = jobshop->create_initial_state(); // Set initial system state
	return jobshop;
}

void report_jobshop_solution(ostream& os, FlexJobShop::JobShop& shop, vector<Action>& actions) {
	FlexJobShop::JobShop::PriorityOrder order;
	shop.get_priority_order(actions, order);
	FlexJobShop::Schedule* sch = shop.build_schedule(order, false);
	os << endl;
	shop.log_schedule(*sch, os);
	os << endl;
	shop.log_gantt(*sch, os);
	delete sch;
}

/*********************************************************/
/* ----------------------- SCRIPT ---------------------- */
/*********************************************************/

void run_sags(string file, int inst, int episodes, int nsim, optional<int> steps, optional<int> nthrds, int trial, int seed, float epsilon, bool with_learning) {
	
	cout << "File -> " << file.c_str();

	/// 1. Create a Simulator
	Sim sim;

	/// 2. Create a JobShop
	FlexJobShop::JobShop* shop = load_jobshop(file, &sim);
	shop->gen = std::mt19937(seed);
	shop->reward = RewardType::incr;

	/// 3. Run SAGS
	SAGS sags(*shop,shop->gen);	
	sags.run(episodes, nsim, steps, epsilon, nthrds, with_learning);
	cout << "\n Runtime=" << sags.wall_time;
	
	/// 4. Outputs
	std::ostringstream result_file;
	result_file << "sags_" << std::setw(2) << std::setfill('0') << inst;
	result_file << "(";
	result_file << episodes << ",";
	result_file << nsim << ",";
	if (steps == nullopt) result_file << "unrestr,";
	else result_file << *steps;
	result_file << ")";
	result_file << "[" << trial << "].txt";

	ofstream out(result_file.str());
	sags.report(out, episodes);
	report_jobshop_solution(out, *shop, sags.best_actions);
	
	/// 5. Tidy up
	delete shop;
}

/*********************************************************/
/* ----------------------- SCRIPT ---------------------- */
/*********************************************************/

void run_actor_critic(string file, optional<int> nthrds, int inst, int episodes, int trial, int seed,
	float lrate, float disc, int rbuff, int nrep, int nroll, optional<int> horiz, bool on_policy) {
	
	cout << "File -> " << file.c_str();
	
	/// 1. Create a Simulator
	Sim sim;
	
	/// 2. Create a JobShop
	FlexJobShop::JobShop* shop = load_jobshop(file, &sim);	
	shop->gen = std::mt19937(seed);
	shop->reward = RewardType::incr; 

	/// 3. Create AC solver and parameterise
	ActorCritic ac(*shop, false);
	ac.episode = episodes;
	ac.alpha_critic = lrate;  
	ac.alpha_actor = 0.1f * ac.alpha_critic;  
	ac.gamma = disc; 
	ac.replay_buffer = rbuff; 
	ac.nreplay = nrep; 
	ac.nroll = nroll;  
	ac.rollout_horizon = horiz;
	ac.on_policy = on_policy;
	ac.nthrds = nthrds;
		
	/// 4. Run AC
	ac.run(episodes);
		
	/// 5. Report Results
	cout << "\nDone. Best Found = " << ac.best_score;	

	/// ac_01(1000,0.01,0.9,100,20,0)[2].txt
	std::ostringstream result_file;
	result_file << "ac_" << std::setw(2) << std::setfill('0') << inst;
	result_file << "(" << ac.episode << ",";
	result_file << ac.alpha_critic << ",";
	result_file << ac.gamma << ",";
	result_file << ac.replay_buffer << ",";
	result_file << ac.nreplay << ",";
	result_file << ac.nroll << ",";
	result_file << *ac.rollout_horizon << ",";
	if (on_policy) result_file << "on_policy";
	else result_file << "off_policy";
	result_file << ")[" << trial << "].txt";

	ofstream out(result_file.str());
	ac.report(out);
	report_jobshop_solution(out, *shop, ac.best_actions);
	
	/// 5. Tidy up
	delete shop;
}

/*********************************************************/
/* ----------------------- MAIN ------------------------ */
/*********************************************************/

int main() {

	int trial = 5;
	int seed = 2; 
	int episodes = 2000;
	int nsim = 50; 
	float epsilon = 0.1f;
	optional<int> steps(1000);
	optional<int> nthrds = 4; 
	bool learn = false;
	int nroll = 10;

	string folder;
	string file = folder + R"(\instances\dauzere\01a.txt)";
	
	run_sags(file, 1, episodes, nsim, steps, nthrds, trial, seed, epsilon, learn);

	//run_actor_critic(file, nthrds, 1, episodes, trial, seed, 0.01, 0.9, 0, 0, nroll, steps, false);
}